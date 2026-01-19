import os
import sys
sys.path.append(os.getcwd())

import json
from pathlib import Path
import random
from itertools import product

import numpy as np

from src.modules.pipeline import Pipeline
from src.modules.bm25 import BM25Module
from src.modules.embeddings import EmbeddingModule
from src.modules.jaccard import JaccardModule
from src.modules.levenshtein import  LevenshteinModule
from src.modules.ner import NERModule


class Evaluator:

    def __init__(self):
        pass


    @staticmethod
    def __validate(y_true: np.ndarray, y_pred: np.ndarray):
        mask = ~np.isnan(y_true) & np.triu(np.ones_like(y_true, dtype=bool), k=1)
        y_true_bin = np.zeros_like(y_pred, dtype=np.int8)
        y_true_bin[mask] = y_true[mask].astype(np.int8)

        TP = float(np.sum(y_pred * y_true_bin * mask))
        FN = float(np.sum((1 - y_pred) * y_true_bin * mask))
        FP = float(np.sum(y_pred * (1 - y_true_bin) * mask))
        TN = float(np.sum((1 - y_pred) * (1 - y_true_bin) * mask))

        return TP, FN, FP, TN


    @staticmethod
    def evaluate(pipeline: Pipeline, X, y):
        TP = FN = FP = TN = 0
        for x, y_true in zip(X, y):
            # Предсказание с выбранными порогами
            y_pred = pipeline.predict(x)
            
            # Подсчет метрик
            TP_tmp, FN_tmp, FP_tmp, TN_tmp = Evaluator.__validate(y_true, y_pred)
            
            TP += TP_tmp
            FN += FN_tmp
            FP += FP_tmp
            TN += TN_tmp

        precision_1 = TP / (TP + FP) if (TP + FP) != 0 else 0.
        recall_1 = TP / (TP + FN) if (TP + FN) != 0 else 0.
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) != 0 else 0.
        support_1 = TP + FN

        precision_0 = TN / (TN + FN)  if (TN + FN) != 0 else 0.
        recall_0 = TN / (TN + FP)  if (TN + FP) != 0 else 0.
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) != 0 else 0.
        support_0 = TN + FP

        return {
            "1": {
                "precision": precision_1,
                "recall": recall_1,
                "f1": f1_1,
                "support": support_1,
            }, 
            "0": {
                "precision": precision_0,
                "recall": recall_0,
                "f1": f1_0,
                "support": support_0,
            },
        }




if __name__ == "__main__":

    def read_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


    def load_daily_data(texts_dir, labels_dir):
        texts_dir = Path(texts_dir)
        labels_dir = Path(labels_dir)

        x, y = [], []  # list[list[str]], list[np.ndarray]
        
        # предполагаем 1к1 соответствие по имени файла
        for text_file in sorted(texts_dir.glob("*.jsonl")):
            label_file = labels_dir / text_file.name
            if not label_file.exists():
                print(f"No matching label file for {text_file.name}")
                continue

            # ---------- texts ----------
            daily_texts = [row["text"] for row in read_jsonl(text_file)]

            x.append(daily_texts)

            k = len(daily_texts)

            # индекс текста → позиция в матрице
            text_to_idx = {text: i for i, text in enumerate(daily_texts)}

            # ---------- labels ----------
            mat = np.full((k, k), np.nan, dtype=float)

            for row in read_jsonl(label_file):
                t1 = row["text_1"]
                t2 = row["text_2"]
                ans = row["answer"]

                if t1 not in text_to_idx or t2 not in text_to_idx:
                    # если вдруг встретилась разметка для текста,
                    # которого нет в texts этого дня
                    continue

                i, j = text_to_idx[t1], text_to_idx[t2]
                mat[i, j] = ans
                mat[j, i] = ans

            y.append(mat)

        return x, y
    

    # Загрузка данных
    x, y = load_daily_data(texts_dir="./dataset/dates", labels_dir="./dataset/responses")
    
    # Разбиение на выборки
    total_objs = len(x)
    random.seed(42)
    train_ids = set(random.sample(list(range(total_objs)), k=total_objs*4//5))
    val_ids = set(range(total_objs)) - train_ids

    train_x, train_y = [x[i] for i in train_ids], [y[i] for i in train_ids]
    val_x, val_y = [x[i] for i in val_ids], [y[i] for i in val_ids]

    # Инициализация модулей
    modules = [
        NERModule(
            model_name="ru_core_news_lg",
            entity_types={"PER", "LOC", "ORG"},
        ), 
        LevenshteinModule(use_quick_filter=False), 
        JaccardModule(n=2, lowercase=True), 
        EmbeddingModule(batch_size=16), 
        BM25Module(),
    ]

    # Перебор всех комбинаций модуль
    config_combs = list(product(range(2), repeat=len(modules)))
    for config in config_combs[1:]:
        model_name = "".join(str(i) for i in config)
        print(f"{model_name=}")

        selected_modules = [modules[i] for i, use in enumerate(config) if use]
        print(f"{selected_modules=}")
        
        pipeline = Pipeline(selected_modules)
        pipeline.fit(train_x, train_y)
        pipeline.save("./configs", model_name="".join(str(i) for i in config))
        print(f"{pipeline.fit_logs=}")

        eval_log = Evaluator.evaluate(pipeline, val_x, val_y)
        print(f"{eval_log=}")

        with open("eval_logs.jsonl", "a", encoding="utf-8") as f:
            eval_log["model"] = model_name
            f.write(json.dumps(eval_log, ensure_ascii=False) + "\n")
