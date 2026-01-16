import json
import pandas as pd

from modules import LevenshteinModule, EmbeddingModule


def test_modules():
    with open("data_source/data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data[:100])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    print("Testing LevenshteinModule...")
    lev_module = LevenshteinModule(threshold=0.5, use_date_filter=False)
    lev_pairs = lev_module.compute_pairs(df)
    print(f"Found {len(lev_pairs)} pairs\n")

    print("Testing predict_pair...")
    score = lev_module.predict_pair(df.loc[0, 'text'], df.loc[1, 'text'])
    print(f"Similarity: {score:.4f}\n")

    print("Testing EmbeddingModule...")
    emb_module = EmbeddingModule(threshold=0.7, use_date_filter=False)
    emb_pairs = emb_module.compute_pairs(df)
    print(f"Found {len(emb_pairs)} pairs\n")

    print("Ok!")


if __name__ == '__main__':
    test_modules()
