from __future__ import annotations

import pytest
import torch

from transformer_book_lab.grouped_query_attention import GroupedQueryAttention
from transformer_book_lab.rope import build_rope_cache

D_MODEL = 64
NUM_HEADS = 8
BATCH = 2
SEQ = 10


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_gqa_groups_set_on_construction():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    assert m.groups == 4


def test_mqa_construction():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=1)
    assert m.groups == NUM_HEADS


def test_invalid_num_kv_heads_raises():
    with pytest.raises(ValueError):
        GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=3)


def test_non_divisible_d_model_raises():
    with pytest.raises(ValueError):
        GroupedQueryAttention(65, NUM_HEADS, num_kv_heads=2)


# ---------------------------------------------------------------------------
# Projection shapes
# ---------------------------------------------------------------------------


def test_kv_projection_shape_gqa():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    head_dim = D_MODEL // NUM_HEADS  # 8
    assert m.k_proj.weight.shape == (2 * head_dim, D_MODEL)  # (16, 64)
    assert m.v_proj.weight.shape == (2 * head_dim, D_MODEL)
    assert m.q_proj.weight.shape == (D_MODEL, D_MODEL)


def test_kv_projection_shape_mqa():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=1)
    head_dim = D_MODEL // NUM_HEADS  # 8
    assert m.k_proj.weight.shape == (head_dim, D_MODEL)
    assert m.v_proj.weight.shape == (head_dim, D_MODEL)


def test_projections_have_no_bias():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert getattr(m, name).bias is None


# ---------------------------------------------------------------------------
# Forward — shapes
# ---------------------------------------------------------------------------


def test_gqa_output_shape():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = m(x, x, x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_mqa_output_shape():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=1)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = m(x, x, x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_mha_equivalent_output_shape():
    m_gqa = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=NUM_HEADS)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = m_gqa(x, x, x)
    assert out.shape == (BATCH, SEQ, D_MODEL)


def test_output_dtype_is_float32():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    x = torch.randn(BATCH, SEQ, D_MODEL)
    out = m(x, x, x)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# RoPE integration
# ---------------------------------------------------------------------------


def test_rope_integration_shape_unchanged():
    m = GroupedQueryAttention(D_MODEL, NUM_HEADS, num_kv_heads=2)
    head_dim = D_MODEL // NUM_HEADS
    x = torch.randn(BATCH, SEQ, D_MODEL)
    cos, sin = build_rope_cache(SEQ, head_dim)
    out = m(x, x, x, cos=cos, sin=sin)
    assert out.shape == (BATCH, SEQ, D_MODEL)
