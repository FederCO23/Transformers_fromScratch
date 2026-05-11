# Learning Log

A running journal of insights, confusions resolved, and decisions made while working through the chapters.

---

## 2026-05-09 — Scaffold created

- Initialized project structure with `pyproject.toml`, `src/`, `tests/`, `notebooks/`, `docs/`, `experiments/`, `openspec/`.
- Defined the OpenSpec change-proposal workflow.
- Pinned the chapter delivery pattern: notebook + module + tests + exercise + optional extension.
- Defined policy: book code is private reference only; all committed code is original.

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

<!-- Add new entries above this line, most recent first. -->
