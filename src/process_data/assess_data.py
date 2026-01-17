import os
import json
from itertools import combinations
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ollama


def embed_batch(texts, batch_size=8):
    with torch.no_grad(), open(p, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            for emb, text in zip(emb.cpu().numpy().tolist(), batch):
                f.write(json.dumps({"text": text, "emb": emb}, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

def classify_pair(pair, system_prompt):
    prompt = (
        system_prompt
        + f'\nText1: "{pair["text_1"]}"\n'
        + f'Text2: "{pair["text_2"]}"\n'
    )

    response = ollama.generate(
        model="qwen3:8b",
        prompt=prompt,
        format={
            "type": "object",
            "properties": {
                "answer": {"type": "integer", "enum": [0, 1]},
                "reason": {"type": "string"}
            },
            "required": ["answer", "reason"]
        },
        options={
            "temperature": 0,
            "num_ctx": 2048,
            "num_predict": 1024,
            "top_k": 20,
            "top_p": 0.8,
            "num_gpu": 999,
            "seed": 42,
        },
    )

    return response["response"]

# Промпты
system_prompt = """
Ты - эксперт по анализу текстов и классификации содержания. 

### ТВОЯ ЗАДАЧА:
1. Проанализировать тексты, выделить ключевые смысловые элементы, затем сделать вывод.
2. Сравнить два текста и определить, имеют ли они одну и ту же суть.
3. Вывод должен быть **валидным JSON**, строго в следующей схеме:
{
  "answer": 1,   # 1 если тексты об одном и том же, 0 если нет
  "reason": "строка с анализом"
}

### ПРАВИЛА ПРИСВОЕНИЯ КЛАССА 1 ("ДУПЛИКАТЫ"):
- Только если тексты ПОЛНОСТЬЮ совпадают по смыслу.
- Описывается одно и то же событие или факт.
- Совпадает основной смысл сообщения, включая причину, цель или последствия события.
- Различия носят только стилистический характер (перефразирование, порядок слов, незначительные уточнения).
- Тексты приводят к одинаковым выводам для читателя.
- Один из текстов полностью содержит в себе всю информацию другого текста.

### ПРАВИЛА ПРИСВОЕНИЯ КЛАССА 0 ("РАЗНЫЕ ТЕКСТЫ"):
- Если нет ПОЛНОГО совпадения смыслового содержания текстов (даже если тексты пересекаются фактами).
- Если тексты описывают одно событие, но делают разные смысловые акценты и приходят к разным выводам.
- Если при совпадении вводной информации они по-разному интерпретируют причину, цель или последствия события и несут различную смысловую нагрузку.

### ОБЩИЕ ВАЖНЫЕ ПРАВИЛА:
- Совпадение факта или события само по себе НЕ является основанием считать тексты дубликатами. Приоритет имеет совпадение причин, целей, последствий и итогового вывода.
- Объясняй свой ответ точно и лаконично.

### ПРИМЕРЫ:
Пример 1:
Text1: "Apple анонсировала новый iPhone 15 на ежегодной презентации."
Text2: "На конференции компания Apple представила iPhone 15."
Вывод:
{
  "answer": 1,
  "reason": "Оба текста сообщают о презентации iPhone 15 компанией Apple. Смысл совпадает."
}

Пример 2:
Text1: "О снижении ставок по ипотеке сообщил Сбербанк. Обновлённые условия направлены на поддержку спроса на жильё и повышение доступности кредитов для населения."
Text2: "Сбербанк сообщил о снижении ставок по ипотеке. Обновлённые условия связаны с пересмотром внутренних подходов к работе с кредитными продуктами и оптимизацией финансовых процессов банка."
Вывод:
{
  "answer": 0,
  "reason": "Тексты не являются дубликатами, т.к. при совпадении вводной информации они несут различную смысловую нагрузку."
}

Пример 3:
Text1: "Погода сегодня ясная, +25 градусов."
Text2: "В Москве сегодня солнечно, температура воздуха достигает +25 градусов, ветер южный 1 м/c."
Вывод:
{
  "answer": 1,
  "reason": "Второй текст полностью покрывает смысл первого и дополняет его."
}

Теперь оцени следующие тексты:
"""

user_prompt = """
Text1: "{text1}"
Text2: "{text2}"
Вывод:
"""

# Все данные
notes = []
for folder in os.listdir("./data"):
    with open(os.path.join("data", folder, "data.json"), "r", encoding="utf-8") as f:
        notes.extend(json.load(f))

df = pd.DataFrame(notes)
df["date"] = pd.to_datetime(df["date"])
del notes

# Все даты
tmp = df.groupby(by="date")["source"].apply(lambda x: len(x.unique())).reset_index()
dates = tmp[tmp["source"] > 1]["date"].values

# Максимум объектов за дату
threshold = 0.75
MAX_OBJS = 50
MAX_TEXTS_REPEATS = 3
TRF = "deepvk/USER-bge-m3"

st = len(os.listdir("./dataset/responses"))
for first_date in dates[st:200]:
    # Дата
    print("Date:", first_date)

    df_date = df[df["date"] == first_date].reset_index(drop=True)
    print(df_date)

    # Векторы
    p = f"./dataset/dates/{first_date}.jsonl"
    if not os.path.exists(p):
        model = SentenceTransformer(TRF).eval()
        embed_batch(df_date["text"].values)
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
    
    embs = []
    with open(p, "r") as f:
        for l in f:
            embs.append(json.loads(l))

    df_date["emb"] = df_date["text"].map({e["text"]: np.array(e["emb"]) for e in embs})

    # Матрица близости
    embs = np.stack(df_date["emb"].to_numpy())
    sim_matrix = embs @ embs.T

    iu = np.triu_indices_from(sim_matrix, k=1)
    vals = sim_matrix[iu]

    # Подозрительные пары
    pairs = []
    texts = df_date["text"].values
    sources = df_date["source"].values
    used = defaultdict(int)
    random.seed(42)
    combs = list(combinations(range(len(texts)), 2))
    random.shuffle(combs)
    for i, j in combs:
        if sources[i] == sources[j]:
            continue
        
        sim = sim_matrix[i, j]
        if sim >= threshold:
            t1, t2 = texts[i], texts[j]

            if (used[t1] == MAX_TEXTS_REPEATS) or (used[t2] == MAX_TEXTS_REPEATS):
                continue

            pairs.append({
                "source_1": sources[i],
                "source_2": sources[j],
                "text_1": t1,
                "text_2": t2,
                "similarity": float(sim)
            })

            used[t1] += 1
            used[t2] += 1
           
            if len(pairs) > MAX_OBJS:
                break

    with open(f"./dataset/pairs/{first_date}.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=4, ensure_ascii=False)

    # Оценка с помощью LLM
    print("count:", len(pairs))
    for pair in tqdm(pairs):
        try:
            model_ans = classify_pair(pair, system_prompt)
            response = json.loads(model_ans)
        except Exception as e:
            if "\"answer\": 1" in model_ans:  # В случае ошибки оставляем положительный класс
                reason = model_ans.split("\"reason\":")[-1].strip(" \"") + "..."
                response = {"answer": 1, "reason": reason}
            else:
                continue

        with open(f"./dataset/responses/{first_date}.jsonl", "a", encoding="utf-8") as f:
            response["text_1"] = pair["text_1"]
            response["text_2"] = pair["text_2"]
            f.write(json.dumps(response, ensure_ascii=False) + "\n")
