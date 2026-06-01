import torch

from transformer_book_lab.relative_position_bias import RelativePositionBias

_NUM_HEADS = 4
_NUM_BUCKETS = 32
_MAX_DIST = 128
_SEQ = 10


def _make() -> RelativePositionBias:
    return RelativePositionBias(
        num_heads=_NUM_HEADS, num_buckets=_NUM_BUCKETS, max_distance=_MAX_DIST
    )


def test_parameter_count() -> None:
    rpb = _make()
    total = sum(p.numel() for p in rpb.parameters())
    assert total == _NUM_BUCKETS * _NUM_HEADS


def test_output_shape() -> None:
    rpb = _make()
    out = rpb(_SEQ)
    assert out.shape == (_NUM_HEADS, _SEQ, _SEQ)


def test_output_dtype() -> None:
    rpb = _make()
    assert rpb(_SEQ).dtype == torch.float32


def test_diagonal_entries_equal() -> None:
    # distance 0 always maps to the same bucket, so diagonal entries per head are equal
    rpb = _make()
    out = rpb(_SEQ)
    for h in range(_NUM_HEADS):
        diag = out[h].diagonal()
        assert diag.eq(diag[0]).all(), f"Head {h} diagonal not uniform"
