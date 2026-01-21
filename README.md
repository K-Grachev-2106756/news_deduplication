# News Deduplication

Модульный пайплайн для поиска дубликатов новостных текстов
## Установка

```bash
pip install -e .
python -m spacy download ru_core_news_lg
```

## Модули

| Тип | Модуль | Описание |
|-----|--------|----------|
| Sparse | Levenshtein | Посимвольное сходство |
| Sparse | BM25 | Частотное сходство терминов |
| Sparse | Jaccard | Триграммы слов + IoU |
| Sparse | NER | Совпадение именованных сущностей |
| Sparse | POS | Совпадение существительных и глаголов |
| Dense | Embeddings | Семантическое сходство (USER-bge-m3) |
| Dense | LSTM | Siamese BiLSTM для попарного сравнения |

## Использование

```python
from src.modules import (
    JaccardModule, LevenshteinModule, BM25Module,
    NERModule, EmbeddingModule, PipelineMean
)

# Создание пайплайна
modules = [JaccardModule(), LevenshteinModule(), NERModule()]
pipeline = PipelineMean(modules)

# Обучение (подбор порога)
pipeline.fit(X_train, y_train)

# Предсказание
predictions = pipeline.predict(texts)
```

## Пайплайны

- **Pipeline** — умножение бинарных предсказаний (AND логика)
- **PipelineMean** — усреднение логитов + один общий порог

## Структура

```
src/
├── modules/
│   ├── base.py          # SparseModule, DenseModule
│   ├── pipeline.py      # Pipeline, PipelineMean
│   ├── jaccard.py
│   ├── levenshtein.py
│   ├── bm25.py
│   ├── ner.py
│   ├── pos.py
│   ├── embeddings.py
│   └── lstm.py
├── eval/
│   └── evaluator.py
└── process_data/
    └── utils.py
```

## Авторы

- Кирилл Грачёв
-  Григорий Орлов