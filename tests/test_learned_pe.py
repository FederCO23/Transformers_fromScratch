import pytest
import torch

from transformer_book_lab.learned_pe import LearnedPositionalEncoding

_D_MODEL = 16
_MAX_SEQ = 32
_BATCH = 2
_SEQ = 10


def _make() -> LearnedPositionalEncoding:
    return LearnedPositionalEncoding(d_model=_D_MODEL, max_seq_len=_MAX_SEQ)


def test_parameter_count() -> None:
    lpe = _make()
    total = sum(p.numel() for p in lpe.parameters())
    assert total == _D_MODEL * _MAX_SEQ


def test_output_shape() -> None:
    lpe = _make()
    x = torch.zeros(_BATCH, _SEQ, _D_MODEL)
    assert lpe(x).shape == (_BATCH, _SEQ, _D_MODEL)


def test_output_dtype() -> None:
    lpe = _make()
    x = torch.zeros(_BATCH, _SEQ, _D_MODEL, dtype=torch.float32)
    assert lpe(x).dtype == torch.float32


def test_overflow_raises_index_error() -> None:
    lpe = _make()
    x = torch.zeros(_BATCH, _MAX_SEQ + 1, _D_MODEL)
    with pytest.raises(IndexError):
        lpe(x)


def test_exact_max_seq_len_is_allowed() -> None:
    lpe = _make()
    x = torch.zeros(_BATCH, _MAX_SEQ, _D_MODEL)
    assert lpe(x).shape == (_BATCH, _MAX_SEQ, _D_MODEL)
