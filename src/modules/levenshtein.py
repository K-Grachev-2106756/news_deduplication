from typing import List

import numpy as np
from Levenshtein import distance as lev_distance
from tqdm.auto import tqdm

from .base import Module


class LevenshteinModule(Module):

    default_threshold = 0.5
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    def __init__(self, use_quick_filter: bool = True):
        self.use_quick_filter = use_quick_filter

    def _similarity(self, text1: str, text2: str) -> float:
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0
        dist = lev_distance(text1, text2) / max_len
        return 1.0 - dist

    def get_logits(self, X: List[str]) -> np.ndarray:
        k = len(X)
        lengths = [len(t) for t in X]

        matrix = np.zeros((k, k), dtype=np.float32)
        for i in tqdm(range(k), desc="Levenshtein"):
            for j in range(i + 1, k):
                if self.use_quick_filter:
                    len_ratio = min(lengths[i], lengths[j]) / max(lengths[i], lengths[j]) if max(lengths[i], lengths[j]) > 0 else 0
                    if len_ratio < 0.5:
                        continue

                    quick_dist = lev_distance(X[i][:100].lower(), X[j][:100].lower()) / 100
                    if quick_dist > 0.7:
                        continue

                sim = self._similarity(X[i], X[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return "LevenshteinModule()"
