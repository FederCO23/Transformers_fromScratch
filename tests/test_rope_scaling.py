from __future__ import annotations

import math

import pytest
import torch

from transformer_book_lab.rope import apply_rope, build_rope_cache
from transformer_book_lab.rope_scaling import (
    build_rope_cache_linear,
    build_rope_cache_ntk,
    build_rope_cache_yarn,
    yarn_attention_scale,
)


# ---------------------------------------------------------------------------
# build_rope_cache_linear
# ---------------------------------------------------------------------------


def test_linear_output_shapes():
    cos, sin = build_rope_cache_linear(seq_len=16, head_dim=8, scale_factor=2.0)
    assert cos.shape == (16, 4)
    assert sin.shape == (16, 4)


def test_linear_output_dtype():
    cos, sin = build_rope_cache_linear(seq_len=8, head_dim=8, scale_factor=2.0)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_linear_device_placement(device):
    cos, sin = build_rope_cache_linear(seq_len=8, head_dim=8, scale_factor=2.0, device=device)
    assert cos.device == device
    assert sin.device == device


def test_linear_identity_at_scale_one():
    cos_l, sin_l = build_rope_cache_linear(seq_len=8, head_dim=8, scale_factor=1.0)
    cos_r, sin_r = build_rope_cache(seq_len=8, head_dim=8)
    assert torch.allclose(cos_l, cos_r)
    assert torch.allclose(sin_l, sin_r)


def test_linear_compresses_frequencies():
    _, sin_scaled = build_rope_cache_linear(seq_len=8, head_dim=8, scale_factor=2.0)
    _, sin_base = build_rope_cache(seq_len=8, head_dim=8)
    # At position 1 all angles are in (0, ~1 rad) so sin is positive and monotonic;
    # halving the frequency halves the angle → smaller sin values.
    assert (sin_scaled[1] < sin_base[1]).all()


def test_linear_odd_head_dim_raises():
    with pytest.raises(ValueError):
        build_rope_cache_linear(seq_len=8, head_dim=7, scale_factor=2.0)


def test_linear_scale_factor_below_one_raises():
    with pytest.raises(ValueError):
        build_rope_cache_linear(seq_len=8, head_dim=8, scale_factor=0.5)


# ---------------------------------------------------------------------------
# build_rope_cache_ntk
# ---------------------------------------------------------------------------


def test_ntk_output_shapes():
    cos, sin = build_rope_cache_ntk(seq_len=16, head_dim=8, scale_factor=2.0)
    assert cos.shape == (16, 4)
    assert sin.shape == (16, 4)


def test_ntk_output_dtype():
    cos, sin = build_rope_cache_ntk(seq_len=8, head_dim=8, scale_factor=2.0)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_ntk_identity_at_scale_one():
    cos_n, sin_n = build_rope_cache_ntk(seq_len=8, head_dim=8, scale_factor=1.0)
    cos_r, sin_r = build_rope_cache(seq_len=8, head_dim=8)
    assert torch.allclose(cos_n, cos_r)
    assert torch.allclose(sin_n, sin_r)


def test_ntk_preserves_high_freq_more_than_linear():
    seq_len, head_dim, scale = 8, 8, 4.0
    cos_base, _ = build_rope_cache(seq_len=seq_len, head_dim=head_dim)
    cos_lin, _ = build_rope_cache_linear(seq_len=seq_len, head_dim=head_dim, scale_factor=scale)
    cos_ntk, _ = build_rope_cache_ntk(seq_len=seq_len, head_dim=head_dim, scale_factor=scale)
    # highest-frequency dim is index 0 (largest theta)
    diff_lin = (cos_lin[:, 0] - cos_base[:, 0]).abs().mean()
    diff_ntk = (cos_ntk[:, 0] - cos_base[:, 0]).abs().mean()
    assert diff_ntk < diff_lin, "NTK should compress high-freq dims less than linear"


def test_ntk_odd_head_dim_raises():
    with pytest.raises(ValueError):
        build_rope_cache_ntk(seq_len=8, head_dim=7, scale_factor=2.0)


def test_ntk_scale_factor_below_one_raises():
    with pytest.raises(ValueError):
        build_rope_cache_ntk(seq_len=8, head_dim=8, scale_factor=0.5)


# ---------------------------------------------------------------------------
# build_rope_cache_yarn
# ---------------------------------------------------------------------------


def test_yarn_output_shapes():
    cos, sin = build_rope_cache_yarn(seq_len=16, head_dim=8, scale_factor=2.0)
    assert cos.shape == (16, 4)
    assert sin.shape == (16, 4)


def test_yarn_output_dtype():
    cos, sin = build_rope_cache_yarn(seq_len=8, head_dim=8, scale_factor=2.0)
    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32


def test_yarn_near_identity_at_scale_one():
    cos_y, _ = build_rope_cache_yarn(seq_len=8, head_dim=8, scale_factor=1.0)
    cos_r, _ = build_rope_cache(seq_len=8, head_dim=8)
    assert (cos_y - cos_r).abs().max().item() < 1e-5


def test_yarn_differs_from_linear_and_ntk():
    cos_y, _ = build_rope_cache_yarn(seq_len=8, head_dim=16, scale_factor=4.0)
    cos_l, _ = build_rope_cache_linear(seq_len=8, head_dim=16, scale_factor=4.0)
    cos_n, _ = build_rope_cache_ntk(seq_len=8, head_dim=16, scale_factor=4.0)
    assert not torch.allclose(cos_y, cos_l), "YaRN should differ from pure linear"
    assert not torch.allclose(cos_y, cos_n), "YaRN should differ from pure NTK"


def test_yarn_odd_head_dim_raises():
    with pytest.raises(ValueError):
        build_rope_cache_yarn(seq_len=8, head_dim=7, scale_factor=2.0)


def test_yarn_scale_factor_below_one_raises():
    with pytest.raises(ValueError):
        build_rope_cache_yarn(seq_len=8, head_dim=8, scale_factor=0.5)


# ---------------------------------------------------------------------------
# yarn_attention_scale
# ---------------------------------------------------------------------------


def test_yarn_attention_scale_at_one():
    assert yarn_attention_scale(1.0) == pytest.approx(1.0)


def test_yarn_attention_scale_at_four():
    expected = 0.1 * math.log(4.0) + 1.0
    assert yarn_attention_scale(4.0) == pytest.approx(expected, rel=1e-6)


def test_yarn_attention_scale_below_one_raises():
    with pytest.raises(ValueError):
        yarn_attention_scale(0.5)


# ---------------------------------------------------------------------------
# apply_rope compatibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("builder", [
    build_rope_cache_linear,
    build_rope_cache_ntk,
    build_rope_cache_yarn,
])
def test_apply_rope_compatibility(builder):
    cos, sin = builder(seq_len=8, head_dim=16, scale_factor=2.0)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == (1, 2, 8, 16)
    assert k_rot.shape == (1, 2, 8, 16)
    assert q_rot.dtype == torch.float32
    assert k_rot.dtype == torch.float32
