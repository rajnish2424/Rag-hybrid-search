"""
Reranker (FlashRank, local ONNX cross-encoder).

Two-stage retrieval: hybrid search casts a WIDE cheap net (num_candidates), then
the reranker does a NARROW precise pass. A cross-encoder reads the query and each
candidate together, so it's far more accurate at judging relevance than the
first-stage vector similarity -- but too slow to run over the whole corpus, which
is exactly why it only runs on the shortlist.
"""
from __future__ import annotations


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._ranker = None

    def _get(self):
        if self._ranker is None:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self.model_name)
        return self._ranker

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        from flashrank import RerankRequest
        passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
        results = self._get().rerank(RerankRequest(query=query, passages=passages))
        out = []
        for r in results[:top_k]:
            c = dict(candidates[r["id"]])
            c["rerank_score"] = round(float(r["score"]), 5)
            out.append(c)
        return out
