from typing import List, Tuple
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from collections import defaultdict

from .base import DuplicationModule


class EmbeddingModule(DuplicationModule):

    def __init__(self, model_name: str = "deepvk/USER-bge-m3", threshold: float = 0.7,
                 batch_size: int = 8, use_date_filter: bool = True, max_length: int = 2000):
        super().__init__("embeddings", threshold)
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_date_filter = use_date_filter
        self.max_length = max_length
        self.model = None
        self.embeddings = None

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _embed_texts(self, texts):
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

    def compute_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        self._load_model()

        truncated_texts = [t[:self.max_length] for t in df['text'].values]
        self.embeddings = self._embed_texts(truncated_texts)

        pairs = []

        if self.use_date_filter and 'month' in df.columns:
            month_to_indices = defaultdict(list)
            for idx, row in df.iterrows():
                month_to_indices[row['month']].append(idx)

            def get_window_indices(month):
                indices = []
                for m in [month - 1, month, month + 1]:
                    indices.extend(month_to_indices.get(m, []))
                return indices

            for month in tqdm(df['month'].unique(), desc="Computing similarities"):
                indices = get_window_indices(month)

                if len(indices) < 2:
                    continue

                window_embeddings = self.embeddings[indices]
                similarities = cosine_similarity(window_embeddings)

                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        sim = similarities[i, j]
                        if sim > self.threshold:
                            pairs.append((indices[i], indices[j], float(sim)))
        else:
            similarities = cosine_similarity(self.embeddings)

            for i in tqdm(range(len(df)), desc="Extracting pairs"):
                for j in range(i + 1, len(df)):
                    sim = similarities[i, j]
                    if sim > self.threshold:
                        pairs.append((i, j, float(sim)))

        return pairs

    def predict_pair(self, text1: str, text2: str) -> float:
        self._load_model()

        text1_truncated = text1[:self.max_length]
        text2_truncated = text2[:self.max_length]

        with torch.no_grad():
            emb1 = self.model.encode([text1_truncated], normalize_embeddings=True)[0]
            emb2 = self.model.encode([text2_truncated], normalize_embeddings=True)[0]

        return float(np.dot(emb1, emb2))
