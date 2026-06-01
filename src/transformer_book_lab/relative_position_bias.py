from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class RelativePositionBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _distance_to_bucket(self, relative_position: Tensor) -> Tensor:
        # T5 convention: half buckets for exact small distances, half log-spaced
        half = self.num_buckets // 2
        ret = (relative_position < 0).long() * half
        n = relative_position.abs()

        max_exact = half // 2
        is_small = n < max_exact
        val_if_large = (
            max_exact
            + (
                torch.log(n.float().clamp(min=max_exact) / max_exact)
                / math.log(self.max_distance / max_exact)
                * (half - max_exact)
            ).long()
        ).clamp(max=half - 1)

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, seq_len: int) -> Tensor:
        positions = torch.arange(seq_len, device=self.embedding.weight.device)
        # relative distances: (seq_len, seq_len) — query row, key col
        relative = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len)
        buckets = self._distance_to_bucket(relative)                 # (seq_len, seq_len)
        bias = self.embedding(buckets)                               # (seq_len, seq_len, num_heads)
        return bias.permute(2, 0, 1)                                 # (num_heads, seq_len, seq_len)
