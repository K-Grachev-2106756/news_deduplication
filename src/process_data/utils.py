import json
from pathlib import Path

import numpy as np


def form_dense_data(X: list[str], y: np.ndarray):
    h, w = y.shape
    xes, yes = [], []
    for i in range(h):
        for j in range(i + 1, w):
            if np.isnan(y[i][j]):
                continue
            xes.append((X[i], X[j]))
            yes.append(int(y[i][j]))
    
    return xes, yes


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
        objs, daily_texts = [], set()
        for row in read_jsonl(label_file):
            objs.append(row)
            daily_texts.add(row["text_1"])
            daily_texts.add(row["text_2"])
        
        daily_texts = sorted(list(daily_texts))  # Для воспроизводимости последовательностей!!! 
        x.append(daily_texts)
        k = len(daily_texts)

        # индекс текста → позиция в матрице
        text_to_idx = {text: i for i, text in enumerate(daily_texts)}

        # ---------- labels ----------
        mat = np.full((k, k), np.nan, dtype=float)
        for row in objs:
            t1 = row["text_1"]
            t2 = row["text_2"]
            ans = row["answer"]

            i, j = text_to_idx[t1], text_to_idx[t2]
            mat[i, j] = ans
            mat[j, i] = ans

        y.append(mat)

    return x, y
