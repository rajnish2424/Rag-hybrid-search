# Hybrid-Search RAG — Pro build

A rebuild of the "local hybrid search RAG" tutorial, upgraded for **running
anywhere on a weak CPU** and for **being evaluated properly**. The heavy LLM runs
on a **hosted API** (Groq, free tier); the light models (embeddings, reranking)
run locally as tiny ONNX models.

## What this adds over the original

| Improvement | Where |
|---|---|
| **Search once, reuse everywhere** (no double search) + query & embedding caches | `rag_app/pipeline.py`, `rag_app/cache.py` |
| **Citations & provenance** — every answer cites numbered sources with page numbers; honest "I don't know" | `rag_app/prompts.py`, `app.py` |
| **Evaluation harness** (retrieval metrics + Ragas generation metrics + reranker ablation) | `eval/` |
| **Ingestion robustness** — page tracking, de-duplication, scanned-PDF detection, index versioning | `rag_app/ingest.py`, `rag_app/store.py` |
| **Structured tracing** — per-stage latency, tokens, what was retrieved/reranked, written to `storage/traces.jsonl` | `rag_app/tracing.py` |
| **Presets instead of model paths** — Fast / Balanced / High-quality | `rag_app/config.py`, `app.py` |
| **Runs anywhere** — hosted LLM + NumPy vector store (no Postgres, no local LLM) | throughout |

## Architecture in one picture

```
PDF ─▶ extract pages ─▶ sentence-aware chunking (+overlap) ─▶ embed (local ONNX) ─▶ NumPy vector index + BM25
                                                                                          │
query ─▶ embed ─▶ HYBRID SEARCH (dense cosine + BM25, fused by RRF) ─▶ rerank (local ONNX) ─▶ top-K contexts
                                                                                          │
                                          RAG prompt (with numbered sources) ─▶ HOSTED LLM (Groq) ─▶ answer + [citations]
```

Only the final generation step leaves your machine. Embeddings and reranking are
small ONNX models — fine on a weak CPU because they aren't doing slow
token-by-token generation.

## Setup

1. **Get a free Groq key** at https://console.groq.com → API Keys.
2. Configure and install:
   ```bash
   cp .env.example .env          # paste your GROQ_API_KEY
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
   Pick a preset, upload a PDF, ask questions. Each answer shows its **Sources**
   and an expandable **Trace**.

> First run downloads the small embedding + reranker models (a few hundred MB total), cached afterward.

## Run it anywhere (no local horsepower)

**Streamlit Community Cloud (free):** push this folder to a public GitHub repo →
share.streamlit.io → "New app" → point at `app.py` → in **Advanced settings →
Secrets**, add:
```
GROQ_API_KEY="your_key"
```
The app then runs in the cloud; your laptop just shows the browser tab. Hugging
Face Spaces (Streamlit SDK) works the same way. Keep documents modest on free
tiers (~1 GB RAM).

## The presets (what the knobs mean)

| | Fast | Balanced | High-quality |
|---|---|---|---|
| LLM (hosted) | Llama-3.1-8B | Llama-3.1-8B | Llama-3.3-70B |
| Embedder (dim) | bge-small (384) | bge-base (768) | bge-large (1024) |
| Reranker | off | MiniLM-L-12 | MiniLM-L-12 |
| Retrieve → context | 8 → 4 | 10 → 5 | 20 → 8 |

Change or add presets in `rag_app/config.py`.

## Evaluation — start here if you've never run one

The mental model: **RAG fails at two independent stages**, so you measure them
separately.

- **Retrieval** — *did we fetch the right chunk?* → `Recall@k`, `MRR` (computed
  transparently in `eval/retrieval_metrics.py`, no framework, no LLM needed).
- **Generation** — *given that context, was the answer right and grounded?* →
  Ragas: `faithfulness` (hallucination check), `answer_relevancy`,
  `context_precision`, `context_recall` (LLM-as-judge).

### Step 1 — build a golden set

A golden item marks the *correct source* by a document name + a short snippet
that must appear in the right chunk:
```json
{"question": "...", "ground_truth": "the correct answer",
 "source_doc": "your.pdf", "source_snippet": "verbatim phrase from the source"}
```
Edit `eval/datasets/golden.jsonl` to match **a PDF you actually indexed**. The
included examples assume an `employee_handbook.pdf` — replace them with your own,
or auto-draft candidates (then review every one by hand):
```bash
python -m eval.build_golden --preset Balanced --n 20
```

### Step 2 — run it

```bash
# retrieval only (fast, no judge/LLM needed):
python -m eval.run_eval --preset Balanced --no-generation

# full eval incl. Ragas generation metrics:
python -m eval.run_eval --preset Balanced

# prove the reranker earns its latency (on vs off):
python -m eval.run_eval --preset Balanced --ablate-rerank
```
Results print to the console and save to `eval/results.json`.

### Step 3 — read the numbers

- **`recall@10_candidates` low?** Your first-stage retrieval is missing the
  answer entirely — fix chunking or the embedder before touching anything else.
  No reranker or prompt can recover a chunk that was never retrieved.
- **`recall@10_candidates` high but `recall@5_final` higher after rerank?**
  That gap *is* the reranker's lift.
- **`faithfulness` low?** The model is asserting things not in the context —
  hallucination. Tighten the prompt or improve retrieval.
- **`context_recall` low?** The right info isn't reaching the LLM — a retrieval
  problem wearing a generation costume.

Then run **ablations** to turn earlier open questions into numbers: sweep
`chunk_size_words` / `chunk_overlap_words` (re-index after changing these),
`num_candidates`, `max_contexts`, and reranker on/off — and keep whatever moves
the metrics.

> **Judge choice matters:** eval uses Llama-3.3-70B as the judge while the product
> may use the 8B model. Your judge should be at least as strong as what it grades.
> Ragas' API changes between versions — this targets `ragas==0.2.10`.

## Layout

```
app.py                     Streamlit UI (presets, upload, chat, citations, trace)
rag_app/
  config.py                Settings + PRESETS + signatures
  chunking.py              sentence-aware chunking with overlap
  embeddings.py            local ONNX embedder (+ optional hosted) with cache
  store.py                 NumPy vector index + BM25 + RRF hybrid + versioning
  rerank.py                FlashRank cross-encoder
  llm.py                   hosted generation via LiteLLM (streaming + token usage)
  pipeline.py              search-once engine, caching, tracing, fallback
  prompts.py               citation-forcing RAG prompt + honest fallback
  cache.py                 embedding + query caches
  tracing.py               per-query structured trace
eval/
  retrieval_metrics.py     Recall@k, MRR (transparent, no deps)
  run_eval.py              runner: retrieval + Ragas + ablation
  build_golden.py          synthetic golden-set drafter (review before use!)
  datasets/golden.jsonl    example golden set (replace with your own)
```

## Honest limitations

- The NumPy store loads all vectors in memory — great for learning and modest
  corpora, not for millions of chunks. Swap in Qdrant/LanceDB/pgvector at scale
  (the `Store` interface is the seam).
- No OCR: scanned PDFs are detected and flagged, not indexed.
- Sentence splitting is regex-based (lightweight, no spaCy download); fine for
  clean prose, weaker on tables/code.
