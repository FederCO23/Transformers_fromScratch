import pytest
import torch

from transformer_book_lab.decoder_layer import DecoderLayer


@pytest.fixture()
def layer(device: torch.device) -> DecoderLayer:
    return DecoderLayer(d_model=64, nhead=4, dim_feedforward=128).to(device)


def test_output_shape_differing_lengths(layer: DecoderLayer, device: torch.device) -> None:
    tgt = torch.randn(2, 8, 64, device=device)
    memory = torch.randn(2, 12, 64, device=device)
    out = layer(tgt, memory)
    assert out.shape == (2, 8, 64)


def test_output_shape_equal_lengths(layer: DecoderLayer, device: torch.device) -> None:
    tgt = torch.randn(2, 6, 64, device=device)
    memory = torch.randn(2, 6, 64, device=device)
    out = layer(tgt, memory)
    assert out.shape == (2, 6, 64)


def test_forward_with_memory_padding_mask(layer: DecoderLayer, device: torch.device) -> None:
    tgt = torch.randn(2, 4, 64, device=device)
    memory = torch.randn(2, 8, 64, device=device)
    mask = torch.zeros(2, 8, dtype=torch.bool, device=device)
    mask[:, 6:] = True  # Last two src positions are padding
    out = layer(tgt, memory, memory_key_padding_mask=mask)
    assert out.shape == (2, 4, 64)


def test_causal_mask_blocks_future_positions(layer: DecoderLayer, device: torch.device) -> None:
    layer.eval()
    tgt = torch.randn(1, 4, 64, device=device)
    tgt_len = tgt.size(1)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), diagonal=1
    )
    with torch.no_grad():
        _, weights = layer.self_attn(tgt, tgt, tgt, attn_mask=causal_mask, need_weights=True)
    # weights: (1, tgt_len, tgt_len) averaged over heads
    # Position 0 must not attend to positions 1, 2, 3
    assert weights[0, 0, 1].item() < 1e-6
    assert weights[0, 0, 2].item() < 1e-6
    assert weights[0, 1, 2].item() < 1e-6
