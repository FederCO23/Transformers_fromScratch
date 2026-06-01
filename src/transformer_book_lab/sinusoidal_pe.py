from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)          # (max_seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                            # (d_model // 2,)
        pe = torch.zeros(max_seq_len, d_model)                      # (max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)
