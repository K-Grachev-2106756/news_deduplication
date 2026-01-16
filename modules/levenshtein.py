from typing import List, Tuple
import pandas as pd
from Levenshtein import distance as lev_distance
from tqdm.auto import tqdm

from .base import DuplicationModule


class LevenshteinModule(DuplicationModule):

    def __init__(self, threshold: float = 0.5, use_date_filter: bool = True):
        super().__init__("levenshtein", threshold)
        self.use_date_filter = use_date_filter

    def compute_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int, float]]:
        pairs = []

        if self.use_date_filter and 'month' in df.columns:
            for month in tqdm(df['month'].unique(), desc="Levenshtein (by month)"):
                mask = (df['month'] >= month - 1) & (df['month'] <= month + 1)
                indices = df[mask].index.tolist()

                if len(indices) < 2:
                    continue

                texts = {idx: df.loc[idx, 'text'] for idx in indices}
                lengths = {idx: len(texts[idx]) for idx in indices}

                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]

                        len_ratio = min(lengths[idx1], lengths[idx2]) / max(lengths[idx1], lengths[idx2])
                        if len_ratio < 0.5:
                            continue

                        text1_start = texts[idx1][:100].lower()
                        text2_start = texts[idx2][:100].lower()
                        quick_dist = lev_distance(text1_start, text2_start) / 100
                        if quick_dist > 0.7:
                            continue

                        sim = self.predict_pair(texts[idx1], texts[idx2])

                        if sim > self.threshold:
                            pairs.append((idx1, idx2, sim))
        else:
            for i in tqdm(range(len(df)), desc="Levenshtein (all pairs)"):
                for j in range(i + 1, len(df)):
                    text1 = df.loc[i, 'text']
                    text2 = df.loc[j, 'text']

                    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
                    if len_ratio < 0.5:
                        continue

                    sim = self.predict_pair(text1, text2)

                    if sim > self.threshold:
                        pairs.append((i, j, sim))

        return pairs

    def predict_pair(self, text1: str, text2: str) -> float:
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 0.0

        dist = lev_distance(text1, text2) / max_len
        return 1.0 - dist
