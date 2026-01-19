from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Module(ABC):

    default_threshold: float = 0.5
    thresholds: List[float] = np.round(np.arange(0.025, 1, step=0.025), 3).tolist()

    @abstractmethod
    def get_logits(self, X: List[str]) -> np.ndarray:
        pass

    def predict(self, X: List[str], threshold: float = None) -> np.ndarray:
        if threshold is None:
            threshold = self.default_threshold
        logits = self.get_logits(X)
        return (logits > threshold).astype(int)

    @classmethod
    def set_thresholds(cls, start: float, end: float, step: float=0.025):
        cls.tr = np.round(np.arange(start, end, step=step), 3).tolist()
