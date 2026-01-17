from typing import List

import numpy as np
import bm25s
import pymorphy3
from tqdm.auto import tqdm

from .base import Module


class BM25Module(Module):

    default_threshold = 0.3
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def _lemmatize(self, text: str) -> List[str]:
        tokens = text.lower().split()
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def get_logits(self, X: List[str]) -> np.ndarray:
        corpus = [self._lemmatize(text) for text in tqdm(X, desc="BM25: lemmatization")]

        retriever = bm25s.BM25()
        retriever.index(corpus)

        k = len(X)
        matrix = np.zeros((k, k), dtype=np.float32)

        for i in tqdm(range(k), desc="BM25: scoring"):
            scores = retriever.get_scores(corpus[i])
            matrix[i, :] = scores

        matrix = (matrix + matrix.T) / 2

        max_score = matrix.max()
        if max_score > 0:
            matrix = matrix / max_score

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return "BM25Module()"
