import pytest
import torch

from transformer_book_lab.rope import apply_rope, build_rope_cache

_SEQ = 8
_HEAD_DIM = 16
_BATCH = 2
_HEADS = 4


def test_cache_shape() -> None:
    cos, sin = build_rope_cache(seq_len=_SEQ, head_dim=_HEAD_DIM)
    assert cos.shape == (_SEQ, _HEAD_DIM // 2)
    assert sin.shape == (_SEQ, _HEAD_DIM // 2)


def test_odd_head_dim_raises() -> None:
    with pytest.raises(ValueError):
        build_rope_cache(seq_len=_SEQ, head_dim=7)


def test_apply_rope_output_shapes() -> None:
    cos, sin = build_rope_cache(_SEQ, _HEAD_DIM)
    q = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM)
    k = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == (_BATCH, _HEADS, _SEQ, _HEAD_DIM)
    assert k_rot.shape == (_BATCH, _HEADS, _SEQ, _HEAD_DIM)


def test_apply_rope_output_dtype() -> None:
    cos, sin = build_rope_cache(_SEQ, _HEAD_DIM)
    q = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM, dtype=torch.float32)
    k = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM, dtype=torch.float32)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.dtype == torch.float32
    assert k_rot.dtype == torch.float32


def test_rotation_is_isometry() -> None:
    cos, sin = build_rope_cache(_SEQ, _HEAD_DIM)
    q = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM)
    k = torch.randn(_BATCH, _HEADS, _SEQ, _HEAD_DIM)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)
