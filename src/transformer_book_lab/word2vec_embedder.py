from __future__ import annotations

import torch
from gensim.models import Word2Vec


class Word2VecEmbedder:
    def __init__(
        self,
        corpus: list[str],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
    ) -> None:
        tokenized = [sentence.split() for sentence in corpus]
        self._model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1,
        )

    def embed(self, word: str) -> torch.Tensor:
        if word not in self._model.wv:
            raise KeyError(word)
        return torch.from_numpy(self._model.wv[word].copy())

    def __contains__(self, word: str) -> bool:
        return word in self._model.wv

    @property
    def vocab_size(self) -> int:
        return len(self._model.wv)
