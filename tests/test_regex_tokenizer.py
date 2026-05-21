from transformer_book_lab.regex_tokenizer import RegexTokenizer


def test_word_boundary_pattern():
    assert RegexTokenizer(r"\w+").tokenize("don't stop!") == ["don", "t", "stop"]


def test_punctuation_splitting():
    assert RegexTokenizer(r"[A-Za-z]+").tokenize("hello, world!") == ["hello", "world"]


def test_no_matches():
    assert RegexTokenizer(r"\d+").tokenize("no digits here") == []


def test_pattern_stored_on_instance():
    t = RegexTokenizer(r"\w+")
    assert t.pattern == r"\w+"
