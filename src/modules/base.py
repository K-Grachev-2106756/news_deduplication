from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class SparseModule(ABC):

    default_threshold: float = 0.5
    thresholds: List[float] = np.round(np.arange(0.025, 1, step=0.025), 3).tolist()
    short_name = "Sparse"

    @abstractmethod
    def get_logits(self, X: List[str]) -> np.ndarray:
        pass

    def predict(self, X: List[str], threshold: float = None) -> np.ndarray:
        if threshold is None:
            threshold = self.default_threshold
        logits = self.get_logits(X)
        return (logits > threshold).astype(int)

    @classmethod
    def set_thresholds(cls, start: float, end: float, step: float = 0.025):
        cls.thresholds = np.round(np.arange(start, end, step=step), 3).tolist()


class DenseModule(SparseModule):

    short_name = "Dense"

    @abstractmethod
    def fit(self, X_pairs: List[Tuple[str, str]], y: List[int]) -> None:
        pass

    def is_fitted(self) -> bool:
        return getattr(self, '_fitted', False)
