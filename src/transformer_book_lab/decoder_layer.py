from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DecoderLayer(nn.Module):
    """Single transformer decoder layer (causal self-attn + cross-attn + feed-forward).

    Uses post-norm placement (LayerNorm after residual addition).
    The causal self-attention mask is generated internally on each forward pass.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # tgt:    (batch, tgt_len, d_model)
        # memory: (batch, src_len, d_model)
        tgt_len = tgt.size(1)
        # True at (i, j) means position i cannot attend to position j
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )  # (tgt_len, tgt_len)

        self_attn_out, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=causal_mask, need_weights=False
        )
        tgt = self.norm1(tgt + self.dropout(self_attn_out))  # (batch, tgt_len, d_model)

        cross_attn_out, _ = self.cross_attn(
            tgt, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))  # (batch, tgt_len, d_model)

        ff_out = self.ff2(self.dropout(self.activation(self.ff1(tgt))))
        tgt = self.norm3(tgt + self.dropout(ff_out))  # (batch, tgt_len, d_model)

        return tgt
