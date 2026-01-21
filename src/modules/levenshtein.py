from typing import List, Optional

import numpy as np
from Levenshtein import distance as lev_distance
from tqdm.auto import tqdm

from .base import SparseModule


class LevenshteinModule(SparseModule):

    default_threshold = 0.5
    thresholds = np.round(np.arange(0.1, 0.7, step=0.05), 3).tolist()

    def __init__(self, use_quick_filter: bool = True):
        self.use_quick_filter = use_quick_filter

    def _similarity(self, text1: str, text2: str, truncation: Optional[int] = None) -> float:
        if truncation:
            text1, text2 = text1[:truncation], text2[:truncation]
        text1, text2 = text1.lower(), text2.lower()

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
                    min_len, max_len = min(lengths[i], lengths[j]), max(lengths[i], lengths[j])
                    len_ratio = min_len / max_len if max_len else 0
                    if len_ratio < 0.5:
                        quick_sim = self._similarity(X[i], X[j], truncation=100)
                        if quick_sim < 0.3:
                            continue

                sim = self._similarity(X[i], X[j])
                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return "LevenshteinModule()"
