from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer_book_lab.rope import apply_rope


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch, seq_q, _ = q.shape
        _, seq_k, _ = k.shape

        q_h = (
            self.q_proj(q)
            .view(batch, seq_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, num_heads, seq_q, head_dim)

        k_h = (
            self.k_proj(k)
            .view(batch, seq_k, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, num_kv_heads, seq_k, head_dim)

        v_h = (
            self.v_proj(v)
            .view(batch, seq_k, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )  # (batch, num_kv_heads, seq_k, head_dim)

        # Expand KV heads to match query head count
        k_h = k_h.repeat_interleave(self.groups, dim=1)  # (batch, num_heads, seq_k, head_dim)
        v_h = v_h.repeat_interleave(self.groups, dim=1)

        if cos is not None and sin is not None:
            q_h, k_h = apply_rope(q_h, k_h, cos, sin)

        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_h, k_h, v_h, attn_mask=attn_mask, dropout_p=dropout_p
        )  # (batch, num_heads, seq_q, head_dim)

        out = out.transpose(1, 2).reshape(batch, seq_q, -1)  # (batch, seq_q, d_model)
        return self.out_proj(out)
