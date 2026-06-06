from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate(head_dim: int, scale_factor: float) -> None:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")


def build_rope_cache_linear(
    seq_len: int,
    head_dim: int,
    scale_factor: float = 1.0,
    base: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) each of shape (seq_len, head_dim // 2) with linear frequency scaling.

    Divides all RoPE frequencies by scale_factor, spreading positions across a wider range.
    """
    _validate(head_dim, scale_factor)
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    theta = theta / scale_factor
    positions = torch.arange(seq_len, device=device).float()  # (seq_len,)
    freqs = torch.outer(positions, theta)                      # (seq_len, half)
    return freqs.cos(), freqs.sin()


def build_rope_cache_ntk(
    seq_len: int,
    head_dim: int,
    scale_factor: float = 1.0,
    base: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) each of shape (seq_len, head_dim // 2) with NTK-aware scaling.

    Replaces base with base * scale_factor^(head_dim / (head_dim - 2)), preserving
    high-frequency (local) components more than linear interpolation does.
    """
    _validate(head_dim, scale_factor)
    half = head_dim // 2
    base_ntk = base * (scale_factor ** (head_dim / (head_dim - 2)))
    theta = 1.0 / (base_ntk ** (torch.arange(0, half, device=device).float() / half))
    positions = torch.arange(seq_len, device=device).float()  # (seq_len,)
    freqs = torch.outer(positions, theta)                      # (seq_len, half)
    return freqs.cos(), freqs.sin()


def build_rope_cache_yarn(
    seq_len: int,
    head_dim: int,
    scale_factor: float = 1.0,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) each of shape (seq_len, head_dim // 2) using YaRN NTK-by-parts.

    Blends linear and NTK frequencies per dimension using a ramp weight derived
    from each dimension's wavelength. High-freq dims lean toward linear; low-freq
    dims lean toward NTK (Peng et al. 2023).
    """
    _validate(head_dim, scale_factor)
    half = head_dim // 2
    i = torch.arange(0, half, device=device).float()

    theta_base = 1.0 / (base ** (i / half))                              # standard
    theta_linear = theta_base / scale_factor                              # linear scaling
    base_ntk = base * (scale_factor ** (head_dim / (head_dim - 2)))
    theta_ntk = 1.0 / (base_ntk ** (i / half))                           # NTK scaling

    # wavelength for each frequency dimension: lambda_i = 2π / theta_i
    wavelength = 2.0 * math.pi / theta_base                              # (half,)
    # ramp: 0 → pure NTK, 1 → pure linear
    r = torch.clamp(
        (wavelength / scale_factor - beta_slow) / (beta_fast - beta_slow), 0.0, 1.0
    )
    theta_yarn = r * theta_linear + (1.0 - r) * theta_ntk

    positions = torch.arange(seq_len, device=device).float()             # (seq_len,)
    freqs = torch.outer(positions, theta_yarn)                            # (seq_len, half)
    return freqs.cos(), freqs.sin()


def yarn_attention_scale(scale_factor: float) -> float:
    """Return the YaRN attention logit multiplier for a given context extension ratio.

    Compensates for entropy increase when attending over an extended position range.
    Formula: 0.1 * ln(scale_factor) + 1.0 (Peng et al. 2023).
    """
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    return 0.1 * math.log(scale_factor) + 1.0
