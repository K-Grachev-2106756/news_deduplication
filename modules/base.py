from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd


class DuplicationModule(ABC):

    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold

    @abstractmethod
    def compute_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        pass

    @abstractmethod
    def predict_pair(self, text1: str, text2: str) -> float:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(threshold={self.threshold})"
