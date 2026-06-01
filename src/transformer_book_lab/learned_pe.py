from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise IndexError(
                f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=x.device)          # (seq_len,)
        x = x + self.embedding(positions)
        return self.dropout(x)
