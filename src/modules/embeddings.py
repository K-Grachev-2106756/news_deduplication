from typing import List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from .base import DenseModule


class EmbeddingModule(DenseModule):

    default_threshold = 0.7
    thresholds = np.round(np.arange(0.4, 1., step=0.05), 3).tolist()

    def __init__(self, model_name: str = "deepvk/USER-bge-m3",
                 batch_size: int = 8, max_length: int = 8192):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self._fitted = True

    def fit(self, X_pairs: List[Tuple[str, str]], y: List[int]) -> None:
        pass

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

    def get_logits(self, X: List[str]) -> np.ndarray:
        self._load_model()
        truncated = [t[:self.max_length] for t in X]
        embeddings = self._embed_texts(truncated)
        matrix = cosine_similarity(embeddings).astype(np.float32)
        return matrix

    def __repr__(self):
        return f"EmbeddingModule(model={self.model_name})"
