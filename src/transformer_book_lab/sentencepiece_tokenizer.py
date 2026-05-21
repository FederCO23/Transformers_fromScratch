from __future__ import annotations

import os
import tempfile

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, corpus: list[str], vocab_size: int) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_path = os.path.join(tmpdir, "corpus.txt")
            model_prefix = os.path.join(tmpdir, "spm")
            with open(corpus_path, "w", encoding="utf-8") as f:
                f.write("\n".join(corpus))
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=1.0,
                model_type="bpe",
            )
            self._model = spm.SentencePieceProcessor()
            self._model.load(model_prefix + ".model")
        # TemporaryDirectory cleaned up; model is loaded into memory

    def encode(self, text: str) -> list[int]:
        return self._model.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._model.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._model.get_piece_size()
