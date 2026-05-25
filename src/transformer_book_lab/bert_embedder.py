from __future__ import annotations

import torch
from transformers import BertModel, BertTokenizer


class BertEmbedder:
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        layer: int = -1,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._layer = layer
        self._tokenizer = BertTokenizer.from_pretrained(model_name)
        self._model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        # hidden_states: tuple of (num_layers+1) tensors each (1, seq_len, d_model)
        hidden = outputs.hidden_states[self._layer]  # (1, seq_len, d_model)
        return hidden[0, 1:-1, :]  # strip [CLS] and [SEP] → (seq_len, d_model)

    @property
    def d_model(self) -> int:
        return self._model.config.hidden_size
