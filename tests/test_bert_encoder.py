import torch
import pytest
from transformer_book_lab.bert_encoder import BertEncoder


@pytest.fixture
def encoder(device: torch.device) -> BertEncoder:
    return BertEncoder(num_layers=4, d_model=64, nhead=4, dim_feedforward=128, dropout=0.0).to(device)


def test_output_list_length(encoder: BertEncoder, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    hidden_states = encoder(x)
    assert len(hidden_states) == 4


def test_each_hidden_state_shape(encoder: BertEncoder, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    hidden_states = encoder(x)
    for hs in hidden_states:
        assert hs.shape == (2, 10, 64)


def test_padding_mask_passthrough(encoder: BertEncoder, device: torch.device) -> None:
    x = torch.randn(2, 10, 64, device=device)
    mask = torch.zeros(2, 10, dtype=torch.bool, device=device)
    hidden_states = encoder(x, key_padding_mask=mask)
    assert all(hs.shape == (2, 10, 64) for hs in hidden_states)


def test_eval_determinism(encoder: BertEncoder, device: torch.device) -> None:
    encoder.eval()
    x = torch.randn(2, 10, 64, device=device)
    with torch.no_grad():
        out1 = encoder(x)[-1]
        out2 = encoder(x)[-1]
    assert torch.allclose(out1, out2)
