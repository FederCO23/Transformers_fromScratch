import torch
import pytest
from transformer_book_lab.gpt2_decoder import GPT2Decoder


@pytest.fixture
def decoder(device: torch.device) -> GPT2Decoder:
    return GPT2Decoder(num_layers=6, d_model=64, nhead=4, dim_feedforward=128, dropout=0.0).to(device)


def test_output_shape(decoder: GPT2Decoder, device: torch.device) -> None:
    tgt = torch.randn(2, 10, 64, device=device)
    out = decoder(tgt)
    assert out.shape == (2, 10, 64)


def test_output_shape_with_memory(decoder: GPT2Decoder, device: torch.device) -> None:
    tgt = torch.randn(2, 8, 64, device=device)
    memory = torch.randn(2, 12, 64, device=device)
    out = decoder(tgt, memory=memory)
    assert out.shape == (2, 8, 64)


def test_output_shape_without_memory(decoder: GPT2Decoder, device: torch.device) -> None:
    tgt = torch.randn(2, 8, 64, device=device)
    out = decoder(tgt, memory=None)
    assert out.shape == (2, 8, 64)
