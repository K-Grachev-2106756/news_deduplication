import os
import sys
sys.path.append(os.getcwd())

import json
import random
from itertools import product

import numpy as np

from src.modules.pipeline import Pipeline
from src.modules.bm25 import BM25Module
from src.modules.embeddings import EmbeddingModule
from src.modules.jaccard import JaccardModule
from src.modules.levenshtein import  LevenshteinModule
from src.modules.ner import NERModule
from src.process_data.utils import load_daily_data


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
