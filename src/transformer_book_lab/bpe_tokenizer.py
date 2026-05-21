from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BpeTokenizer:
    def __init__(self, corpus: list[str], vocab_size: int) -> None:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, initial_alphabet=ByteLevel.alphabet())
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
