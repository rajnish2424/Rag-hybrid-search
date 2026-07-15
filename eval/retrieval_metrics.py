"""
Retrieval metrics, computed transparently (no framework needed).

We can't know chunk IDs when authoring the golden set (chunks are created at
ingest time), so a golden item marks the CORRECT source by (source_doc,
source_snippet). A retrieved chunk counts as relevant if it comes from that doc
AND contains the (normalized) snippet. From that we get:

- Recall@k : did a relevant chunk appear in the top-k?  (Did retrieval find it?)
- MRR      : 1 / rank of the first relevant chunk.       (How high did it rank?)

Computing these for the pre-rerank candidates vs the post-rerank contexts shows
the reranker's LIFT.
"""
from __future__ import annotations
import re


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def is_relevant(chunk: dict, source_doc: str, source_snippet: str) -> bool:
    if _norm(chunk["doc_name"]) != _norm(source_doc):
        return False
    return _norm(source_snippet) in _norm(chunk["text"])


def first_relevant_rank(ranked_chunks: list[dict], source_doc: str, snippet: str) -> int | None:
    for rank, ch in enumerate(ranked_chunks, start=1):
        if is_relevant(ch, source_doc, snippet):
            return rank
    return None


def recall_at_k(ranked_chunks: list[dict], source_doc: str, snippet: str, k: int) -> float:
    r = first_relevant_rank(ranked_chunks[:k], source_doc, snippet)
    return 1.0 if r is not None else 0.0


def reciprocal_rank(ranked_chunks: list[dict], source_doc: str, snippet: str) -> float:
    r = first_relevant_rank(ranked_chunks, source_doc, snippet)
    return 1.0 / r if r else 0.0


def aggregate(rows: list[dict]) -> dict:
    """rows: per-question dicts with the keys below. Returns dataset means."""
    n = max(1, len(rows))
    keys = ["recall@5_candidates", "recall@5_final", "recall@10_candidates",
            "mrr_candidates", "mrr_final"]
    return {k: round(sum(r.get(k, 0.0) for r in rows) / n, 4) for k in keys}
