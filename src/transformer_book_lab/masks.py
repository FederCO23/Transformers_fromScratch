from __future__ import annotations

import torch
from torch import Tensor


def generate_causal_mask(seq_len: int, device: torch.device | str = "cpu") -> Tensor:
    """Return a boolean upper-triangular causal mask of shape (seq_len, seq_len).

    Position [i, j] is True when j > i, meaning token i cannot attend to token j
    (future positions are masked out in attention).
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )  # (seq_len, seq_len)
