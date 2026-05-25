from __future__ import annotations

import gensim.downloader
import torch
from gensim.models import KeyedVectors


class GloveEmbedder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50") -> None:
        self._kv: KeyedVectors = gensim.downloader.load(model_name)

    def embed(self, word: str) -> torch.Tensor:
        if word not in self._kv:
            raise KeyError(word)
        return torch.from_numpy(self._kv[word].copy())

    def __contains__(self, word: str) -> bool:
        return word in self._kv

    @property
    def vector_size(self) -> int:
        return self._kv.vector_size
