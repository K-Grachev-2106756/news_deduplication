from typing import List, Tuple
import pandas as pd
import bm25s
import pymorphy3
from tqdm.auto import tqdm
from collections import defaultdict

from .base import DuplicationModule


class BM25Module(DuplicationModule):

    def __init__(self, threshold: float = 100.0, use_date_filter: bool = True):
        super().__init__("bm25", threshold)
        self.use_date_filter = use_date_filter
        self.retriever = None
        self.corpus = None
        self.morph = pymorphy3.MorphAnalyzer()

    def _lemmatize(self, text: str) -> List[str]:
        tokens = text.lower().split()
        lemmas = [self.morph.parse(token)[0].normal_form for token in tokens]
        return lemmas

    def compute_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        self.corpus = [self._lemmatize(text) for text in tqdm(df['text'], desc="Lemmatization")]

        self.retriever = bm25s.BM25()
        self.retriever.index(self.corpus)

        pairs = []

        if self.use_date_filter and 'month' in df.columns:
            month_to_indices = defaultdict(set)
            for idx, row in df.iterrows():
                month_to_indices[row['month']].add(idx)

            def get_window_indices(month):
                indices = set()
                for m in [month - 1, month, month + 1]:
                    indices.update(month_to_indices.get(m, set()))
                return indices

            for idx in tqdm(range(len(self.corpus)), desc="BM25 pairs"):
                query = self.corpus[idx]
                scores = self.retriever.get_scores(query)

                month = df.loc[idx, 'month']
                window_indices = get_window_indices(month)

                for idx2 in window_indices:
                    if idx2 > idx and scores[idx2] > self.threshold:
                        pairs.append((idx, idx2, float(scores[idx2])))
        else:
            for idx in tqdm(range(len(self.corpus)), desc="BM25 pairs"):
                query = self.corpus[idx]
                scores = self.retriever.get_scores(query)

                for idx2 in range(idx + 1, len(self.corpus)):
                    if scores[idx2] > self.threshold:
                        pairs.append((idx, idx2, float(scores[idx2])))

        return pairs

    def predict_pair(self, text1: str, text2: str) -> float:
        if self.retriever is None:
            return 0.0

        query = self._lemmatize(text1)
        scores = self.retriever.get_scores(query)

        return float(scores.max()) / 1000.0
