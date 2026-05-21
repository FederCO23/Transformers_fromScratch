from __future__ import annotations


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()
