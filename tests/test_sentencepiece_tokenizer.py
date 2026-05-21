import pytest

from transformer_book_lab.sentencepiece_tokenizer import SentencePieceTokenizer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "transformers are powerful models for natural language processing",
    "sentence piece learns a subword vocabulary from raw text corpora",
    "neural networks learn representations from data and gradient descent",
    "attention mechanisms allow tokens to communicate across the full sequence",
    "self attention computes query key and value projections for every token",
    "layer normalisation stabilises training by normalising all hidden states",
    "residual connections allow gradients to flow through very deep networks",
]

_VOCAB_SIZE = 50


@pytest.fixture(scope="module")
def sp() -> SentencePieceTokenizer:
    return SentencePieceTokenizer(corpus=CORPUS, vocab_size=_VOCAB_SIZE)


def test_trains_without_error(sp: SentencePieceTokenizer) -> None:
    assert sp is not None


def test_encode_returns_ints(sp: SentencePieceTokenizer) -> None:
    ids = sp.encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_round_trip(sp: SentencePieceTokenizer) -> None:
    text = "the quick brown fox"
    assert sp.decode(sp.encode(text)) == text


def test_vocab_size(sp: SentencePieceTokenizer) -> None:
    assert sp.vocab_size == _VOCAB_SIZE
