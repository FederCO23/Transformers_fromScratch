from __future__ import annotations

import pytest
import torch

from transformer_book_lab.multi_head_attention import MultiHeadAttention
from transformer_book_lab.rope import build_rope_cache

D_MODEL = 64
NUM_HEADS = 4
BATCH = 2
SEQ = 10


@pytest.fixture()
def mha() -> MultiHeadAttention:
    return MultiHeadAttention(D_MODEL, NUM_HEADS, dropout=0.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_head_dim_set_on_construction():
    m = MultiHeadAttention(d_model=64, num_heads=4)
    assert m.head_dim == 16


def test_non_divisible_d_model_raises():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=65, num_heads=4)


def test_projections_have_no_bias(mha):
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert getattr(mha, name).bias is None


# ---------------------------------------------------------------------------
# Forward — shapes
# ---------------------------------------------------------------------------


def test_self_attention_output_shapes(mha):
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, weights = mha(x, x, x)
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ, SEQ)


def test_cross_attention_output_shapes(mha):
    q = torch.randn(BATCH, 6, D_MODEL)
    kv = torch.randn(BATCH, SEQ, D_MODEL)
    out, weights = mha(q, kv, kv)
    assert out.shape == (BATCH, 6, D_MODEL)
    assert weights.shape == (BATCH, NUM_HEADS, 6, SEQ)


def test_output_dtype_is_float32(mha):
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out, weights = mha(x, x, x)
    assert out.dtype == torch.float32
    assert weights.dtype == torch.float32


# ---------------------------------------------------------------------------
# Attention weight properties
# ---------------------------------------------------------------------------


def test_attention_weights_sum_to_one(mha):
    x = torch.randn(BATCH, SEQ, D_MODEL)
    _, weights = mha(x, x, x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH, NUM_HEADS, SEQ), atol=1e-5)


# ---------------------------------------------------------------------------
# RoPE integration
# ---------------------------------------------------------------------------


def test_rope_integration_shape_unchanged(mha):
    x = torch.randn(BATCH, SEQ, D_MODEL)
    cos, sin = build_rope_cache(SEQ, mha.head_dim)
    out, weights = mha(x, x, x, cos=cos, sin=sin)
    assert out.shape == (BATCH, SEQ, D_MODEL)
    assert weights.shape == (BATCH, NUM_HEADS, SEQ, SEQ)


# ---------------------------------------------------------------------------
# Additive mask
# ---------------------------------------------------------------------------


def test_additive_mask_zeros_out_positions(mha):
    x = torch.randn(BATCH, SEQ, D_MODEL)
    # Mask the last 5 key positions with a large negative value
    mask = torch.zeros(SEQ, SEQ)
    mask[:, -5:] = -1e9
    _, weights = mha(x, x, x, attn_mask=mask)
    assert weights[..., -5:].max().item() < 1e-6
