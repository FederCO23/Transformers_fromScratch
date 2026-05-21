from transformer_book_lab.whitespace_tokenizer import WhitespaceTokenizer


def test_single_sentence():
    assert WhitespaceTokenizer().tokenize("hello world") == ["hello", "world"]


def test_extra_whitespace_collapsed():
    assert WhitespaceTokenizer().tokenize("  hello   world  ") == ["hello", "world"]


def test_empty_string():
    assert WhitespaceTokenizer().tokenize("") == []


def test_single_token():
    assert WhitespaceTokenizer().tokenize("transformer") == ["transformer"]
