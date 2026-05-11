from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class EncoderLayer(nn.Module):
    """Single transformer encoder layer (self-attn + feed-forward).

    Supports post-norm (original "Attention Is All You Need") and pre-norm
    (modern default) via the ``pre_norm`` flag.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        # x: (batch, seq_len, d_model)
        if self.pre_norm:
            normed = self.norm1(x)
            attn_out, _ = self.self_attn(
                normed, normed, normed,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(attn_out)
            ff_in = self.norm2(x)
            ff_out = self.ff2(self.dropout(self.activation(self.ff1(ff_in))))
            x = x + self.dropout(ff_out)
        else:
            attn_out, _ = self.self_attn(
                x, x, x,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.ff2(self.dropout(self.activation(self.ff1(x))))
            x = self.norm2(x + self.dropout(ff_out))
        return x  # (batch, seq_len, d_model)
