import os
import json
from itertools import product
from typing import Union

import numpy as np

from .base import SparseModule, DenseModule


class Pipeline:

    def __init__(self, module_list: list[Union[SparseModule, DenseModule]]):
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


    def _prepare_thresholds(self, all_logits):
        min_thresholds, max_thresholds = [], []

        for module, x_logits in zip(self.modules, all_logits):
            min_thr = float("-inf")
            max_thr = float("inf")
            for logits in x_logits:
                for thr in module.thresholds:
                    pred = (logits > thr).astype(int)
                    s = pred.sum()
                    if s == 0:
                        max_thr = min(max_thr, thr)
                    if s == pred.size:
                        min_thr = max(min_thr, thr)
                        
            min_thresholds.append(min_thr)
            max_thresholds.append(max_thr)

        return min_thresholds, max_thresholds
    

    def fit(self, X, y):
        # Grid Search
        all_logits = [[module.get_logits(x) for module in self.modules] for x in X]  # Собираем логиты один раз сразу
        
        min_thresholds, max_thresholds = self._prepare_thresholds(all_logits)
        thresholds = [
            [t for t in module.thresholds if min_t < t < max_t]
            for module, min_t, max_t in zip(self.modules, min_thresholds, max_thresholds)
        ]  # Сокращаем количество комбинаций порогов

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


class PipelineMean(Pipeline):

    def __init__(self, module_list: list[Union[SparseModule, DenseModule]]):
        self.modules = module_list
        self.best_param = 0.5
        self.fit_logs = []

    
    def get_logits(self, X: list[str]):
        logits = np.zeros(shape=(len(X), len(X)))
        for module in self.modules:
            logits += module.get_logits(X)
        logits /= len(self.modules)

        return logits


    def predict(self, X: list[str]):
        return (self.get_logits(X) > self.best_param).astype(int)
    

    def fit(self, X, y):
        # Grid Search
        all_logits = [self.get_logits(x) for x in X]  # Собираем логиты один раз сразу
        
        best_score = 0.
        for threshold in np.arange(0, 1, 0.025):
            
            TP = FN = FP = TN = 0
            for x_logits, y_true in zip(all_logits, y):
                # Предсказание с выбранными порогами
                y_pred = (x_logits > threshold).astype(int)
                
                # Подсчет метрик
                TP_tmp, FN_tmp, FP_tmp, TN_tmp = Pipeline._Pipeline__validate(y_true, y_pred)
                
                TP += TP_tmp
                FN += FN_tmp
                FP += FP_tmp
                TN += TN_tmp

            precision = TP / (TP + FP) if (TP + FP) != 0 else 0.
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0.
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.

            # Сохранение результата
            if f1 > best_score:
                self.best_param = threshold
                best_score = f1
            
            self.fit_logs.append({
                "params": threshold, 
                "precision": precision, 
                "recall": recall, 
                "f1": f1,
            })


    def save(self, output_path: str, model_name: str = "model"):
        os.makedirs(output_path, exist_ok=True)
        
        with open(f"{output_path}/{model_name}.json", "w", encoding="utf-8") as f:
            json.dump({
                "modules": [str(module) for module in self.modules], 
                "params": self.best_param,
                "history": self.fit_logs,
            }, f, indent=4, ensure_ascii=False)
