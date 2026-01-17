from typing import List, Set, Tuple

import numpy as np
import spacy
from tqdm.auto import tqdm

from .base import Module


class NERModule(Module):

    default_threshold = 0.4
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    def __init__(self, model_name: str = "ru_core_news_lg",
                 entity_types: Tuple[str, ...] = ("PER", "LOC", "ORG")):
        self.model_name = model_name
        self.entity_types = entity_types
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.model_name)

    def _extract_entities(self, text: str) -> Set[str]:
        doc = self.nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                normalized = ent.text.lower().strip()
                if normalized:
                    entities.add(normalized)
        return entities

    def get_logits(self, X: List[str]) -> np.ndarray:
        self._load_model()
        k = len(X)

        entities_list = [self._extract_entities(text) for text in tqdm(X, desc="NER: extracting entities")]

        matrix = np.zeros((k, k), dtype=np.float32)
        for i in tqdm(range(k), desc="NER: computing similarity"):
            for j in range(i + 1, k):
                ents_i, ents_j = entities_list[i], entities_list[j]
                intersection = len(ents_i & ents_j)
                max_size = max(len(ents_i), len(ents_j))
                sim = intersection / max_size if max_size > 0 else 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return f"NERModule(model={self.model_name})"
