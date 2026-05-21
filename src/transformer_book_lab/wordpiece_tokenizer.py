from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer


class WordPieceTokenizer:
    def __init__(self, corpus: list[str], vocab_size: int) -> None:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.decoder = WordPieceDecoder(prefix="##")
        trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]"])
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
