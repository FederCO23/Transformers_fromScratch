import pytest

from transformer_book_lab.bpe_tokenizer import BpeTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "transformers are powerful models for natural language processing",
    "byte pair encoding merges the most frequent pairs of characters",
    "neural networks learn representations from data and gradient descent",
    "attention mechanisms allow tokens to communicate across the sequence",
    "self attention computes query key and value projections for each token",
    "layer normalisation stabilises training by normalising hidden states",
    "residual connections allow gradients to flow through deep networks",
]

_VOCAB_SIZE = 300


@pytest.fixture(scope="module")
def bpe() -> BpeTokenizer:
    return BpeTokenizer(corpus=CORPUS, vocab_size=_VOCAB_SIZE)


def test_trains_without_error(bpe: BpeTokenizer) -> None:
    assert bpe is not None


def test_encode_returns_ints(bpe: BpeTokenizer) -> None:
    ids = bpe.encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_round_trip(bpe: BpeTokenizer) -> None:
    text = "the quick brown fox"
    assert bpe.decode(bpe.encode(text)) == text


def test_vocab_size(bpe: BpeTokenizer) -> None:
    assert bpe.vocab_size == _VOCAB_SIZE
