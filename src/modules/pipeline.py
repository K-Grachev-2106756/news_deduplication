import os
import json
from itertools import product

import numpy as np

from .base import Module


class Pipeline:

    def __init__(self, module_list: list[Module]):
        # Инициализация модулей
        self.modules = module_list
        self.best_params = [None] * len(module_list)
        self.fit_logs = []
    

    def predict(self, X: list[str]):
        ans = np.ones(shape=(len(X), len(X)))
        for module, threshold in zip(self.modules, self.best_params):
            ans = ans * module.predict(X, threshold)
        
        return ans
    

    @staticmethod
    def __validate(y_true: np.ndarray, y_pred: np.ndarray):
        mask = ~np.isnan(y_true) & np.triu(np.ones_like(y_true, dtype=bool), k=1)
        y_true_bin = np.zeros_like(y_pred, dtype=np.int8)
        y_true_bin[mask] = y_true[mask].astype(np.int8)

        TP = np.sum(y_pred * y_true_bin * mask)
        FN = np.sum((1 - y_pred) * y_true_bin * mask)
        FP = np.sum(y_pred * (1 - y_true_bin) * mask)
        TN = np.sum((1 - y_pred) * (1 - y_true_bin) * mask)

        return TP, FN, FP, TN


    def fit(self, X, y):
        # Grid Search
        thresholds = [module.thresholds for module in self.modules]
        all_logits = [[module.get_logits(x) for module in self.modules] for x in X]  # Собираем логиты один раз сразу
        best_score = 0.
        for params_comb in product(*thresholds):
            
            TP = FN = FP = TN = 0
            for x_logits, y_true in zip(all_logits, y):
                # Предсказание с выбранными порогами
                y_pred = np.ones_like(y_true)
                for logits, threshold in zip(x_logits, params_comb):
                    y_pred = y_pred * (logits > threshold).astype(int)
                
                # Подсчет метрик
                TP_tmp, FN_tmp, FP_tmp, TN_tmp = self.__validate(y_true, y_pred)
                
                TP += TP_tmp
                FN += FN_tmp
                FP += FP_tmp
                TN += TN_tmp

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0.
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0.
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.

            # Сохранение результата
            if f1 > best_score:
                self.best_params = params_comb
                best_score = f1
            
            self.fit_logs.append({
                "params": list(params_comb), 
                "precision": precision, 
                "recall": recall, 
                "f1": f1,
            })


    def save(self, output_path: str, model_name: str = "model"):
        os.makedirs(output_path, exist_ok=True)
        
        with open(f"{output_path}/{model_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "modules": [str(module) for module in self.modules], 
                "params": self.best_params,
                "history": self.fit_logs,
            }, f, indent=4, ensure_ascii=False)
