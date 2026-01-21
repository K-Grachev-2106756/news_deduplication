from typing import List, Set

import numpy as np
import spacy
from tqdm.auto import tqdm

from .base import SparseModule


class POSModule(SparseModule):
    # Part of Speech

    default_threshold = 0.4
    thresholds = np.round(np.arange(0.1, 0.7, step=0.05), 3).tolist()
    short_name = "POS"

    VALID_POS = {"NOUN", "PROPN", "VERB"}

    def __init__(self, model_name: str = "ru_core_news_lg"):
        self.model_name = model_name
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.model_name, disable=["ner"])

    def _extract_terms_batch(self, X: List[str]) -> List[Set[str]]:
        terms_list: List[Set[str]] = []

        for doc in tqdm(
            self.nlp.pipe(X, batch_size=32, n_process=2),
            total=len(X),
            desc="POS: extracting nouns & verbs",
        ):
            terms = set()
            for token in doc:
                if (
                    token.pos_ in self.VALID_POS
                    and not token.is_stop
                    and token.lemma_
                ):
                    lemma = token.lemma_.lower().strip()
                    if lemma:
                        terms.add(lemma)
            terms_list.append(terms)

        return terms_list

    def get_logits(self, X: List[str]) -> np.ndarray:
        self._load_model()

        k = len(X)

        terms_list = self._extract_terms_batch(X)

        matrix = np.zeros((k, k), dtype=np.float32)

        for i in tqdm(range(k), desc="POS: computing similarity"):
            for j in range(i + 1, k):
                t_i, t_j = terms_list[i], terms_list[j]

                intersection = len(t_i & t_j)
                union = len(t_i | t_j)
                sim = intersection / union if union else 0.0

                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def repr(self):
        return f"POSModule(model_name={self.model_name})"
