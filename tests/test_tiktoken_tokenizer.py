from transformer_book_lab.tiktoken_tokenizer import TiktokenTokenizer


def test_constructs_without_args() -> None:
    assert TiktokenTokenizer() is not None


def test_encode_returns_ints() -> None:
    ids = TiktokenTokenizer().encode("hello world")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_encode_is_deterministic() -> None:
    t = TiktokenTokenizer()
    assert t.encode("hello world") == t.encode("hello world")


def test_round_trip() -> None:
    t = TiktokenTokenizer()
    text = "hello world"
    assert t.decode(t.encode(text)) == text


def test_vocab_size() -> None:
    assert TiktokenTokenizer().vocab_size == 100277
