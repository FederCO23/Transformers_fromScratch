from unittest.mock import patch

import numpy as np
import pytest
import torch
from gensim.models import KeyedVectors

from transformer_book_lab.glove_embedder import GloveEmbedder

_VECTOR_SIZE = 10
_WORDS = ["hello", "world", "the", "quick"]


def _make_tiny_kv() -> KeyedVectors:
    kv = KeyedVectors(vector_size=_VECTOR_SIZE)
    vectors = np.random.rand(len(_WORDS), _VECTOR_SIZE).astype(np.float32)
    kv.add_vectors(_WORDS, vectors)
    return kv


@pytest.fixture
def glove() -> GloveEmbedder:
    with patch("gensim.downloader.load", return_value=_make_tiny_kv()):
        yield GloveEmbedder("glove-wiki-gigaword-50")


def test_embed_shape_and_dtype(glove: GloveEmbedder) -> None:
    vec = glove.embed("hello")
    assert vec.shape == (_VECTOR_SIZE,)
    assert vec.dtype == torch.float32


def test_embed_raises_for_oov(glove: GloveEmbedder) -> None:
    with pytest.raises(KeyError):
        glove.embed("xyzzy_oov_word")


def test_contains_known_word(glove: GloveEmbedder) -> None:
    assert "hello" in glove


def test_contains_unknown_word(glove: GloveEmbedder) -> None:
    assert "xyzzy_oov_word" not in glove


def test_vector_size(glove: GloveEmbedder) -> None:
    assert glove.vector_size == _VECTOR_SIZE


@pytest.mark.slow
def test_real_glove_loads() -> None:
    ge = GloveEmbedder("glove-wiki-gigaword-50")
    assert ge.vector_size == 50
    assert "the" in ge
    vec = ge.embed("the")
    assert vec.shape == (50,)
    assert vec.dtype == torch.float32
