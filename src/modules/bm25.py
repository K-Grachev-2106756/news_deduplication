from typing import List

import numpy as np
import bm25s
import pymorphy3
from tqdm.auto import tqdm
import re

from .base import SparseModule


class BM25Module(SparseModule):

    default_threshold = 0.5
    thresholds = np.round(np.arange(0.025, 0.7, step=0.025), 3).tolist()

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()

    def _lemmatize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [self.morph.parse(token)[0].normal_form for token in tokens]

    def get_logits(self, X: List[str]) -> np.ndarray:
        corpus = [self._lemmatize(text) for text in tqdm(X, desc="BM25: lemmatization")]

        retriever = bm25s.BM25()
        retriever.index(corpus)

        k = len(X)
        matrix = np.zeros((k, k), dtype=np.float32)

        for i in tqdm(range(k), desc="BM25: scoring"):
            scores = retriever.get_scores(corpus[i])
            matrix[i, :] = scores

        max_score = matrix.max()
        if max_score > 0:
            matrix = matrix / matrix.max(axis=1)  # bm25 несимметрична, нормализуем по строкам
            matrix = np.maximum(matrix, matrix.T)  # В контексте задачи нам неважно i похож на j или j на i

        return matrix

    def __repr__(self):
        return "BM25Module()"
