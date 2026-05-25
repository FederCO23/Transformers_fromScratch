from unittest.mock import MagicMock, patch

import pytest
import torch

from transformer_book_lab.bert_embedder import BertEmbedder

_D_MODEL = 32
_SEQ_LEN = 4  # includes [CLS] and [SEP], so embed output is (2, _D_MODEL)
_NUM_LAYERS = 13  # bert-base has 12 encoder layers + 1 embedding layer


def _make_mocks() -> tuple[MagicMock, MagicMock]:
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.ones(1, _SEQ_LEN, dtype=torch.long),
        "attention_mask": torch.ones(1, _SEQ_LEN, dtype=torch.long),
    }

    hidden = torch.randn(1, _SEQ_LEN, _D_MODEL)
    mock_output = MagicMock()
    mock_output.hidden_states = tuple(hidden for _ in range(_NUM_LAYERS))

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.config.hidden_size = _D_MODEL

    return mock_tokenizer, mock_model


@pytest.fixture
def bert() -> BertEmbedder:
    mock_tokenizer, mock_model = _make_mocks()
    with patch("transformer_book_lab.bert_embedder.BertTokenizer") as MockTok, \
         patch("transformer_book_lab.bert_embedder.BertModel") as MockModel:
        MockTok.from_pretrained.return_value = mock_tokenizer
        MockModel.from_pretrained.return_value = mock_model
        yield BertEmbedder("bert-base-uncased")


def test_embed_shape(bert: BertEmbedder) -> None:
    vec = bert.embed("hello world")
    assert vec.ndim == 2
    assert vec.shape == (_SEQ_LEN - 2, _D_MODEL)


def test_embed_dtype(bert: BertEmbedder) -> None:
    vec = bert.embed("hello world")
    assert vec.dtype == torch.float32


def test_d_model(bert: BertEmbedder) -> None:
    assert bert.d_model == _D_MODEL


@pytest.mark.slow
def test_real_bert_d_model() -> None:
    be = BertEmbedder()
    assert be.d_model == 768


@pytest.mark.slow
def test_real_bert_embed_shape() -> None:
    be = BertEmbedder()
    vec = be.embed("hello world")
    assert vec.shape == (2, 768)
    assert vec.dtype == torch.float32


@pytest.mark.slow
def test_real_bert_context_sensitivity() -> None:
    be = BertEmbedder()
    v1 = be.embed("I went to the river bank")
    v2 = be.embed("I went to the bank")
    # "bank" is the last content token in both sentences
    assert not torch.allclose(v1[-1], v2[-1])
