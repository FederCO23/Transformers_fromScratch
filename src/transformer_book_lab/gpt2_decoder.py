from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from transformer_book_lab.decoder_layer import DecoderLayer


class GPT2Decoder(nn.Module):
    """Stack of N DecoderLayer blocks (GPT-2-style decoder-only architecture).

    Causal masking is handled internally by each DecoderLayer. Accepts optional
    encoder memory for cross-attention so the same module can serve as the decoder
    half of an encoder-decoder model without API changes. When memory=None,
    each DecoderLayer receives tgt as its memory argument (self-contained).
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor | None = None,
    ) -> Tensor:
        # tgt:    (batch, tgt_len, d_model)
        # memory: (batch, src_len, d_model) or None
        mem = memory if memory is not None else tgt
        x = tgt
        for layer in self.layers:
            x = layer(x, mem)
        return x  # (batch, tgt_len, d_model)
