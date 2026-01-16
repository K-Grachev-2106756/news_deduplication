from typing import List, Tuple
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm.auto import tqdm

from .base import DuplicationModule


class CrossEncoderModule(DuplicationModule):

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                 threshold: float = 0.5, batch_size: int = 128):
        super().__init__("crossencoder", threshold)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def compute_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        raise NotImplementedError("Кросс-энокдер только для переранжирования")

    def predict_pair(self, text1: str, text2: str) -> float:
        self._load_model()
        score = self.model.predict([[text1, text2]])[0]
        return float(score)

    def rerank_pairs(self, pairs: List[Tuple[int, int, float]], df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        self._load_model()

        text_pairs = []
        pair_ids = []

        for idx1, idx2, old_score in pairs:
            text_pairs.append([str(df.loc[idx1, 'text']), str(df.loc[idx2, 'text'])])
            pair_ids.append((idx1, idx2))

        scores = self.model.predict(text_pairs, batch_size=self.batch_size, show_progress_bar=True)

        reranked_pairs = []
        for i, (idx1, idx2) in enumerate(pair_ids):
            if scores[i] > self.threshold:
                reranked_pairs.append((idx1, idx2, float(scores[i])))

        return reranked_pairs
