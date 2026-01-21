import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from .base import DenseModule


class EmbeddingModule(DenseModule):

    default_threshold = 0.7
    thresholds = np.round(np.arange(0.4, 1., step=0.05), 3).tolist()

    def __init__(
            self, 
            model_name: str = "deepvk/USER-bge-m3",
            batch_size: int = 8, 
            max_length: int = 8192,
            lr: float = 2e-5,
            epochs: int = 1,
            warmup_ratio: float = 0.1,
        ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.epochs = epochs
        self.warmup_ratio = warmup_ratio

        self.model: SentenceTransformer | None = None
        self._fitted = False

    def fit(self, X_pairs: List[Tuple[str, str]], y: List[int]) -> None:
        self._load_model()

        train_examples = [
            InputExample(texts=[t1[:self.max_length], t2[:self.max_length]], label=float(label))
            for (t1, t2), label in zip(X_pairs, y)
        ]

        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )

        train_loss = losses.CosineSimilarityLoss(self.model)

        warmup_steps = int(
            len(train_dataloader) * self.epochs * self.warmup_ratio
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": self.lr},
            show_progress_bar=True,
        )

        self._fitted = True

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self.model.eval()

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
                batch = texts[i:i + self.batch_size]
                emb = self.model.encode(
                    batch,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                embeddings.append(emb)
                if (i // self.batch_size) % 10 == 0:
                    torch.cuda.empty_cache()
        return np.vstack(embeddings)

    def get_logits(self, X: list[str]) -> np.ndarray:
        self._load_model()

        truncated = [t[:self.max_length] for t in X]
        embeddings = self._embed_texts(truncated)

        return cosine_similarity(embeddings).astype(np.float32)

    def save(self, output_folder: str):
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Nothing to save.")

        os.makedirs(output_folder, exist_ok=True)

        self.model.save(output_folder)

        # сохраняем метаданные модуля
        meta = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }

        torch.save(meta, os.path.join(output_folder, "module_meta.pt"))

    def load(self, model_folder: str):
        meta_path = os.path.join(model_folder, "module_meta.pt")

        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            self.model_name = meta.get("model_name", self.model_name)
            self.batch_size = meta.get("batch_size", self.batch_size)
            self.max_length = meta.get("max_length", self.max_length)

        self.model = SentenceTransformer(model_folder)
        self.model.eval()

        self._fitted = True

    def __repr__(self):
        return f"EmbeddingModule(model={self.model_name})"
