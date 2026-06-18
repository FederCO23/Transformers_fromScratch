from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer_book_lab.rope import apply_rope


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        def reshape(x: Tensor, seq: int) -> Tensor:
            return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            # -> (batch, num_heads, seq, head_dim)

        q_h = reshape(self.q_proj(q), seq_q)
        k_h = reshape(self.k_proj(k), seq_k)
        v_h = reshape(self.v_proj(v), seq_k)

        if cos is not None and sin is not None:
            q_h, k_h = apply_rope(q_h, k_h, cos, sin)

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, seq_q, seq_k)

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        if self.training and self.dropout > 0.0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        out = torch.matmul(attn_weights, v_h)                       # (batch, heads, seq_q, head_dim)
        out = out.transpose(1, 2).reshape(batch, seq_q, -1)         # (batch, seq_q, d_model)
        return self.out_proj(out), attn_weights
