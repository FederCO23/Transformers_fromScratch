from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer_book_lab.rope import apply_rope


class MultiLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kv_latent_dim: int,
        q_latent_dim: int | None = None,
        rope_head_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_latent_dim = kv_latent_dim
        self.rope_head_dim = rope_head_dim if rope_head_dim is not None else self.head_dim // 2
        self.dropout = dropout

        # KV compression path: x → c_kv → k_c, v
        self.w_dkv = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.w_uk = nn.Linear(kv_latent_dim, num_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(kv_latent_dim, num_heads * self.head_dim, bias=False)

        # Decoupled RoPE branches: applied to separate k_r and q_r projections from x
        # (RoPE cannot be applied after the low-rank projection without breaking its geometry)
        self.w_kr = nn.Linear(d_model, num_heads * self.rope_head_dim, bias=False)
        self.w_qr = nn.Linear(d_model, num_heads * self.rope_head_dim, bias=False)

        # Query path: optional down-projection before up-projection
        if q_latent_dim is not None:
            self.w_dq: nn.Linear | None = nn.Linear(d_model, q_latent_dim, bias=False)
            self.w_uq = nn.Linear(q_latent_dim, d_model, bias=False)
        else:
            self.w_dq = None
            self.w_uq = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor | None = None,
        sin: Tensor | None = None,
    ) -> Tensor:
        batch, seq, _ = x.shape

        # KV latent and up-projections
        c_kv = self.w_dkv(x)                                                       # (B, S, kv_latent_dim)
        k_c = (
            self.w_uk(c_kv)
            .view(batch, seq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, S, head_dim)
        v = (
            self.w_uv(c_kv)
            .view(batch, seq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, S, head_dim)

        # Decoupled RoPE branches
        k_r = (
            self.w_kr(x)
            .view(batch, seq, self.num_heads, self.rope_head_dim)
            .transpose(1, 2)
        )  # (B, H, S, rope_head_dim)
        q_r = (
            self.w_qr(x)
            .view(batch, seq, self.num_heads, self.rope_head_dim)
            .transpose(1, 2)
        )  # (B, H, S, rope_head_dim)

        if cos is not None and sin is not None:
            q_r, k_r = apply_rope(q_r, k_r, cos, sin)

        # Query compressed path
        q_in = self.w_uq(self.w_dq(x)) if self.w_dq is not None else self.w_uq(x)
        q_c = (
            q_in.view(batch, seq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, H, S, head_dim)

        # Concatenate compressed and RoPE dims along head_dim
        q_full = torch.cat([q_c, q_r], dim=-1)   # (B, H, S, head_dim + rope_head_dim)
        k_full = torch.cat([k_c, k_r], dim=-1)   # (B, H, S, head_dim + rope_head_dim)

        # V keeps head_dim; SDPA output is (B, H, S, head_dim)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q_full, k_full, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).reshape(batch, seq, -1)  # (B, S, d_model)
        return self.out_proj(out)
