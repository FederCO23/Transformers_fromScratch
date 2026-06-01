from __future__ import annotations

import torch
from torch import Tensor


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) each of shape (seq_len, head_dim // 2)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()        # (seq_len,)
    freqs = torch.outer(positions, theta)                            # (seq_len, half)
    return freqs.cos(), freqs.sin()


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Rotate q and k by RoPE frequencies.

    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim // 2)
    """
    def rotate(x: Tensor, c: Tensor, s: Tensor) -> Tensor:
        # split into even/odd pairs along last dim
        x1 = x[..., 0::2]   # (batch, heads, seq_len, half)
        x2 = x[..., 1::2]
        # broadcast cos/sin: (1, 1, seq_len, half)
        c = c.unsqueeze(0).unsqueeze(0)
        s = s.unsqueeze(0).unsqueeze(0)
        rotated = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        # interleave back: (batch, heads, seq_len, head_dim)
        return rotated.flatten(-2)

    return rotate(q, cos, sin), rotate(k, cos, sin)
