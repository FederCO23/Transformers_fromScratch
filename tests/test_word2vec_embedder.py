import pytest
import torch

from transformer_book_lab.word2vec_embedder import Word2VecEmbedder

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "transformers are powerful models for natural language processing",
    "word embeddings capture semantic meaning in dense vector space",
    "neural networks learn representations from raw text data",
    "attention mechanisms let every token attend to every other token",
]


@pytest.fixture(scope="module")
def embedder() -> Word2VecEmbedder:
    return Word2VecEmbedder(corpus=CORPUS, vector_size=16, window=3, min_count=1)


def test_trains_without_error(embedder: Word2VecEmbedder) -> None:
    assert embedder is not None


def test_embed_shape_and_dtype(embedder: Word2VecEmbedder) -> None:
    vec = embedder.embed("the")
    assert vec.shape == (16,)
    assert vec.dtype == torch.float32


def test_embed_raises_for_oov(embedder: Word2VecEmbedder) -> None:
    with pytest.raises(KeyError):
        embedder.embed("xyzzy_oov_word")


def test_contains_known_word(embedder: Word2VecEmbedder) -> None:
    assert "the" in embedder


def test_contains_unknown_word(embedder: Word2VecEmbedder) -> None:
    assert "xyzzy_oov_word" not in embedder


def test_vocab_size_positive(embedder: Word2VecEmbedder) -> None:
    assert embedder.vocab_size > 0
