from .base import DuplicationModule
from .levenshtein import LevenshteinModule
from .bm25 import BM25Module
from .embeddings import EmbeddingModule
from .crossencoder import CrossEncoderModule

__all__ = [
    'DuplicationModule',
    'LevenshteinModule',
    'BM25Module',
    'EmbeddingModule',
    'CrossEncoderModule'
]
