from __future__ import annotations

import tiktoken


class TiktokenTokenizer:
    def __init__(self) -> None:
        self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab
