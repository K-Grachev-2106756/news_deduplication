from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Module(ABC):

    default_threshold: float = 0.5
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    @abstractmethod
    def get_logits(self, X: List[str]) -> np.ndarray:
        pass

    def predict(self, X: List[str], threshold: float = None) -> np.ndarray:
        if threshold is None:
            threshold = self.default_threshold
        logits = self.get_logits(X)
        return (logits > threshold).astype(int)
