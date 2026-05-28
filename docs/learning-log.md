# Learning Log

A running journal of insights, confusions resolved, and decisions made while working through the chapters.

---

## 2026-05-09 — Scaffold created

- Initialized project structure with `pyproject.toml`, `src/`, `tests/`, `notebooks/`, `docs/`, `experiments/`, `openspec/`.
- Defined the OpenSpec change-proposal workflow.
- Pinned the chapter delivery pattern: notebook + module + tests + exercise + optional extension.
- Defined policy: book code is private reference only; all committed code is original.

---

## 2026-05-18 — Ch 03: Tokenizers

- Implemented six tokenization strategies — whitespace, regex, BPE, WordPiece, SentencePiece,
  and tiktoken — each in its own module with a consistent interface per strategy tier.
- Rule-based tokenizers (`WhitespaceTokenizer`, `RegexTokenizer`) return `list[str]` and need no
  vocabulary; they are effectively stateless.
- Library-backed tokenizers (`BpeTokenizer`, `WordPieceTokenizer`, `SentencePieceTokenizer`,
  `TiktokenTokenizer`) expose `encode(str) -> list[int]` / `decode(list[int]) -> str` and a
  `vocab_size` property.
- BPE needed `ByteLevel(add_prefix_space=False)` for a clean encode→decode round-trip; with the
  default `add_prefix_space=True` the decoder prepends a spurious space to the first token.
- SentencePiece writes a `.model` file to disk during training; wrapping the trainer inside a
  `tempfile.TemporaryDirectory` context manager keeps the filesystem clean — the model is loaded
  into memory before the directory is deleted.
- tiktoken's `cl100k_base` has exactly 100 277 tokens and requires no training; it is the most
  production-realistic tokenizer in this chapter and the closest to what current LLMs use.
- Key insight from the comparison table: a larger vocabulary encodes the same text in *fewer*
  tokens. tiktoken uses ~40% fewer tokens than the small trained BPE for the probe sentence
  because its vocabulary is 300× larger and contains many whole-word tokens.

---

## 2026-05-10 — Ch 01: Introduction to Attention and Transformers

- Implemented `EncoderLayer` with post-norm and pre-norm variants (single `pre_norm: bool` flag).
  Pre-norm normalises the input *before* the sublayer; post-norm normalises *after* the residual.
  Pre-norm is the modern default (GPT-2, LLaMA) because the raw residual path makes gradient flow cleaner.
- Implemented `DecoderLayer` with causal self-attention (upper-triangular bool mask, generated
  internally), cross-attention from encoder memory, and a post-norm feed-forward sublayer.
- Both modules use `nn.MultiheadAttention(batch_first=True)` as a placeholder — attention internals
  are deferred to Ch 07–09.
- The decoder has ~1.5× more parameters than the encoder at the same `d_model` because of the
  extra cross-attention block (confirmed by parameter count in the notebook).
- Key insight: the encoder is bidirectional (no mask); the decoder is causal (upper-triangular mask).
  Cross-attention is the bridge — decoder queries the encoder's memory at every decoding step.
- Open questions: how does the causal mask interact with multi-head weights individually (before
  averaging)? Ch 07 will shed light on this when we implement MHA from scratch.

## 2026-05-25 — Ch 04: Word Embeddings

- Implemented three embedding families: `Word2VecEmbedder` (gensim skip-gram), `GloveEmbedder`
  (gensim.downloader pre-trained), and `BertEmbedder` (HuggingFace last-layer hidden states).
- Static embedders (`embed(word: str) -> Tensor` of shape `(d,)`) raise `KeyError` on OOV words and
  expose `__contains__` for membership tests. Contextual embedder returns `(seq_len, d_model)` after
  stripping the `[CLS]` and `[SEP]` special tokens.
- All embedders return `torch.Tensor`; numpy arrays from gensim are wrapped with `.copy()` to avoid
  non-writable memory issues before passing to `torch.from_numpy`.
- Test strategy: fast tests mock gensim and HuggingFace backends with in-memory objects so no network
  access is needed in CI; slow tests (`@pytest.mark.slow`) exercise real downloads and are opt-in.
- The "bank" context-sensitivity demo makes the static vs contextual distinction concrete: GloVe
  returns the same vector regardless of context; BERT's cosine similarity between *river bank* and
  *financial bank* is noticeably below 1.0, proving the representations differ.
- Key insight: static embeddings collapse polysemy — a word with multiple meanings gets one averaged
  point in vector space. Contextual embeddings solve this by conditioning on the full sequence, which
  is the core capability that makes fine-tuned language models so powerful.
- Analogy arithmetic (`king - man + woman ≈ queen`) works in both GloVe and Word2Vec because the
  direction of "royalty" and "gender" are consistently encoded across the co-occurrence statistics.

<!-- Add new entries above this line, most recent first. -->
