import pytest
import torch

from transformer_book_lab.encoder_layer import EncoderLayer


@pytest.fixture()
def layer_postnorm(device: torch.device) -> EncoderLayer:
    return EncoderLayer(d_model=64, nhead=4, dim_feedforward=128, pre_norm=False).to(device)


@pytest.fixture()
def layer_prenorm(device: torch.device) -> EncoderLayer:
    return EncoderLayer(d_model=64, nhead=4, dim_feedforward=128, pre_norm=True).to(device)


def test_postnorm_output_shape(layer_postnorm: EncoderLayer, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    out = layer_postnorm(x)
    assert out.shape == (2, 10, 64)


def test_prenorm_output_shape(layer_prenorm: EncoderLayer, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    out = layer_prenorm(x)
    assert out.shape == (2, 10, 64)


def test_forward_with_padding_mask(layer_postnorm: EncoderLayer, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    # Last two positions are padding
    mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
    mask[:, 8:] = True
    out = layer_postnorm(x, src_key_padding_mask=mask)
    assert out.shape == (2, 10, 64)


def test_forward_without_mask_no_exception(
    layer_prenorm: EncoderLayer, device: torch.device
) -> None:
    x = torch.randn(1, 5, 64, device=device)
    out = layer_prenorm(x)
    assert out.shape == (1, 5, 64)
