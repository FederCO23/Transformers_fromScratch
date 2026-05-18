from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from transformer_book_lab.encoder_layer import EncoderLayer


class BertEncoder(nn.Module):
    """Stack of N EncoderLayer blocks (BERT-style encoder-only architecture).

    Returns all intermediate hidden states so downstream tasks can select any layer.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pre_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout, pre_norm)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> list[Tensor]:
        # x: (batch, seq_len, d_model)
        hidden_states: list[Tensor] = []
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
            hidden_states.append(x)
        return hidden_states  # each: (batch, seq_len, d_model)
