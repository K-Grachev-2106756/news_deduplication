from typing import List, Set, Iterable

import numpy as np
import spacy
from tqdm.auto import tqdm

from .base import SparseModule


class NERModule(SparseModule):

    default_threshold = 0.4
    thresholds = np.round(np.arange(0.1, 0.6, step=0.05), 3).tolist()
    short_name = "NER"

    def __init__(self, model_name: str = "ru_core_news_lg",
                 entity_types: Iterable[str] = {"PER", "LOC", "ORG"}):
        self.model_name = model_name
        self.entity_types = entity_types
        self.nlp = None

    def _load_model(self):
        if self.nlp is None:
            self.nlp = spacy.load(self.model_name)
    
    def _extract_entities(self, doc: spacy.tokens.Doc) -> Set[str]:
        entities = set()
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                lemmas = [token.lemma_.lower() for token in ent]
                normalized = " ".join(lemmas).strip()
                if normalized:
                    entities.add(normalized)
        return entities

    def get_logits(self, X: List[str]) -> np.ndarray:
        self._load_model()

        k = len(X)

        # ---- NER extraction via pipe ----
        entities_list = []
        for doc in tqdm(
            self.nlp.pipe(X, batch_size=32, n_process=2, disable=["tagger", "parser"]),
            desc="NER: extracting entities",
            total=k,
        ):
            entities_list.append(self._extract_entities(doc))

        # ---- similarity matrix ----
        matrix = np.zeros((k, k), dtype=np.float32)
        for i in tqdm(range(k), desc="NER: computing similarity"):
            for j in range(i + 1, k):
                ents_i, ents_j = entities_list[i], entities_list[j]
                intersection = len(ents_i & ents_j)
                union = len(ents_i | ents_j)
                sim = intersection / union if union else 0.0
                matrix[i, j] = sim
                matrix[j, i] = sim

        np.fill_diagonal(matrix, 1.0)
        return matrix

    def __repr__(self):
        return f"NERModule(model_name={self.model_name})"
