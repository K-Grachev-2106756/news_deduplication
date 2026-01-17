import re
from typing import List, Set

import numpy as np
from tqdm.auto import tqdm

from .base import Module


class JaccardModule(Module):

    default_threshold = 0.3
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, n: int = 3, lowercase: bool = True):
        self.n = n
        self.lowercase = lowercase

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def _get_ngrams(self, text: str) -> Set[tuple]:
        words = self._tokenize(text)
        if len(words) < self.n:
            return {tuple(words)} if words else set()
        return {tuple(words[i:i+self.n]) for i in range(len(words) - self.n + 1)}

    def get_logits(self, X: List[str]) -> np.ndarray:
        k = len(X)
        ngrams_list = [self._get_ngrams(text) for text in tqdm(X, desc="Jaccard: extracting ngrams")]

        matrix = np.zeros((k, k), dtype=np.float32)
        for i in tqdm(range(k), desc="Jaccard: computing similarity"):
            for j in range(i + 1, k):
                ng_i, ng_j = ngrams_list[i], ngrams_list[j]
                intersection = len(ng_i & ng_j)
                union = len(ng_i | ng_j)
                sim = intersection / union if union > 0 else 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return f"JaccardModule(n={self.n})"
