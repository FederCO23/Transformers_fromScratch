from __future__ import annotations

import re


class RegexTokenizer:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    def tokenize(self, text: str) -> list[str]:
        return re.findall(self.pattern, text)
