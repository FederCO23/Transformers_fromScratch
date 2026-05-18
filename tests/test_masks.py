import torch
import pytest
from transformer_book_lab.masks import generate_causal_mask


def test_shape(device: torch.device) -> None:
    mask = generate_causal_mask(5, device)
    assert mask.shape == (5, 5)


def test_upper_triangle_content() -> None:
    mask = generate_causal_mask(4, "cpu")
    for i in range(4):
        for j in range(4):
            if j > i:
                assert mask[i, j].item() is True
            else:
                assert mask[i, j].item() is False


def test_device_placement(device: torch.device) -> None:
    mask = generate_causal_mask(8, device)
    assert mask.device.type == device.type
