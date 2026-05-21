import pytest

from transformer_book_lab.wordpiece_tokenizer import WordPieceTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "transformers are powerful models for natural language processing",
    "word piece tokenization splits words into subword units for coverage",
    "neural networks learn representations from data and gradient descent",
    "attention mechanisms allow tokens to communicate across the sequence",
    "self attention computes query key and value projections for each token",
    "layer normalisation stabilises training by normalising hidden states",
    "residual connections allow gradients to flow through deep networks",
]


@pytest.fixture(scope="module")
def wp() -> WordPieceTokenizer:
    return WordPieceTokenizer(corpus=CORPUS, vocab_size=200)


def test_trains_without_error(wp: WordPieceTokenizer) -> None:
    assert wp is not None


def test_encode_returns_ints(wp: WordPieceTokenizer) -> None:
    ids = wp.encode("hello")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_round_trip(wp: WordPieceTokenizer) -> None:
    text = "the quick brown fox"
    assert wp.decode(wp.encode(text)) == text


def test_vocab_size_is_positive_int(wp: WordPieceTokenizer) -> None:
    assert isinstance(wp.vocab_size, int)
    assert wp.vocab_size > 0
