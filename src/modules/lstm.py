import os
from typing import List, Tuple, Optional
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from .base import DenseModule


class SiameseLSTM(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_size: int = 128, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers,
                           batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        return torch.cat([h[-2], h[-1]], dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = self.encode(x1)
        e2 = self.encode(x2)
        combined = torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=1)
        return self.classifier(combined).squeeze(-1)


class PairDataset(Dataset):

    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int],
                 word2idx: dict, max_length: int):
        self.pairs = pairs
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def _tokenize(self, text: str) -> List[int]:
        words = text.lower().split()[:self.max_length]
        ids = [self.word2idx.get(w, 1) for w in words]  # 1 = UNK
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))  # 0 = PAD
        return ids

    def __getitem__(self, idx):
        t1, t2 = self.pairs[idx]
        return (
            torch.tensor(self._tokenize(t1)),
            torch.tensor(self._tokenize(t2)),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


class LSTMModule(DenseModule):

    default_threshold = 0.5
    thresholds = np.round(np.arange(0.3, 0.8, step=0.05), 3).tolist()
    short_name = "LSTM"

    def __init__(self, hidden_size: int = 128, embed_dim: int = 128,
                 max_length: int = 256, epochs: int = 10, batch_size: int = 32,
                 min_word_freq: int = 2, device: Optional[str] = None):
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.min_word_freq = min_word_freq
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.word2idx = None
        self._fitted = False

    def _build_vocab(self, texts: List[str]) -> dict:
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in counter.items():
            if freq >= self.min_word_freq:
                word2idx[word] = len(word2idx)
        return word2idx

    def fit(self, X_pairs: List[Tuple[str, str]], y: List[int]) -> None:
        all_texts = [t for pair in X_pairs for t in pair]
        self.word2idx = self._build_vocab(all_texts)

        self.model = SiameseLSTM(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size
        ).to(self.device)

        dataset = PairDataset(X_pairs, y, self.word2idx, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for x1, x2, labels in tqdm(loader, desc=f"LSTM Epoch {epoch+1}/{self.epochs}"):
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x1, x2)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        self.model.eval()
        self._fitted = True

    def _tokenize(self, text: str) -> torch.Tensor:
        words = text.lower().split()[:self.max_length]
        ids = [self.word2idx.get(w, 1) for w in words]
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))
        return torch.tensor(ids).unsqueeze(0).to(self.device)

    def get_logits(self, X: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LSTMModule not fitted")

        n = len(X)
        matrix = np.zeros((n, n), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(n), desc="LSTM"):
                for j in range(i + 1, n):
                    x1 = self._tokenize(X[i])
                    x2 = self._tokenize(X[j])
                    prob = self.model(x1, x2).item()
                    matrix[i, j] = prob
                    matrix[j, i] = prob

        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def save(self, output_folder: str, model_name: str = "lstm"):
        torch.save({
            "word2idx": self.word2idx,
            "max_length": self.max_length,
            "embed_dim": self.embed_dim,
            "hidden_size": self.hidden_size,
            "model_state": self.model.state_dict(),
        }, os.path.join(output_folder, f"{model_name}.pt"))

    def load(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.word2idx = checkpoint["word2idx"]
        self.max_length = checkpoint["max_length"]
        self.embed_dim = checkpoint["embed_dim"]
        self.hidden_size = checkpoint["hidden_size"]

        self.model = SiameseLSTM(
            vocab_size=len(self.word2idx),
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self._fitted = True

    def __repr__(self):
        return f"LSTMModule(hidden_size={self.hidden_size})"
