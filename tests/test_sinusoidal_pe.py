import torch

from transformer_book_lab.sinusoidal_pe import SinusoidalPositionalEncoding

_D_MODEL = 16
_MAX_SEQ = 64
_BATCH = 2
_SEQ = 10


def _make() -> SinusoidalPositionalEncoding:
    return SinusoidalPositionalEncoding(d_model=_D_MODEL, max_seq_len=_MAX_SEQ)


def test_no_trainable_parameters() -> None:
    spe = _make()
    assert len(list(spe.parameters())) == 0


def test_buffer_registered() -> None:
    spe = _make()
    assert hasattr(spe, "pe")
    assert spe.pe.shape == (_MAX_SEQ, _D_MODEL)


def test_output_shape() -> None:
    spe = _make()
    x = torch.zeros(_BATCH, _SEQ, _D_MODEL)
    out = spe(x)
    assert out.shape == (_BATCH, _SEQ, _D_MODEL)


def test_output_dtype_preserved() -> None:
    spe = _make()
    x = torch.zeros(_BATCH, _SEQ, _D_MODEL, dtype=torch.float32)
    assert spe(x).dtype == torch.float32


def test_zero_input_returns_encoding() -> None:
    spe = _make()
    x = torch.zeros(_BATCH, _SEQ, _D_MODEL)
    out = spe(x)
    expected = spe.pe[:_SEQ].unsqueeze(0).expand(_BATCH, -1, -1)
    assert torch.allclose(out, expected)


def test_deterministic() -> None:
    spe1 = SinusoidalPositionalEncoding(d_model=_D_MODEL, max_seq_len=_MAX_SEQ)
    spe2 = SinusoidalPositionalEncoding(d_model=_D_MODEL, max_seq_len=_MAX_SEQ)
    assert torch.allclose(spe1.pe, spe2.pe)
