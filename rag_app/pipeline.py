"""
The RAG engine. This ties everything together and fixes the original tutorial's
"search twice" flaw: we retrieve ONCE, cache it, then reuse those exact chunks
for generation AND for evaluation. No hidden second search.

Public surface:
    engine = RagEngine(settings)
    engine.ingest_pdf(path, name)
    result = engine.retrieve(query)                 # -> RetrieveResult (search-once, cached)
    for ev in engine.answer_stream(query): ...      # -> streams answer + citations + trace
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os

from .config import Settings, DATA_DIR
from .cache import EmbeddingCache, QueryCache
from .embeddings import Embedder
from .store import Store
from .rerank import Reranker
from .ingest import ingest_pdf, IngestReport
from .prompts import RAG_SYSTEM_PROMPT, FALLBACK_SYSTEM_PROMPT, build_context_block
from .llm import stream_completion
from .tracing import QueryTrace


@dataclass
class RetrieveResult:
    query: str
    candidates: list[dict]      # fused hybrid results (pre-rerank)
    contexts: list[dict]        # final chunks sent to the LLM (post-rerank)
    trace: QueryTrace
    grounded: bool


class RagEngine:
    def __init__(self, settings: Settings, preset_name: str = "custom", data_dir: str = DATA_DIR):
        self.s = settings
        self.preset_name = preset_name
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.emb_cache = EmbeddingCache(data_dir, settings.embedder_signature())
        self.embedder = Embedder(settings, self.emb_cache)
        self.store = Store(data_dir, settings.embed_dim, settings.embedder_signature())
        self.store.load()
        self.reranker = Reranker(settings.reranker_model) if settings.use_reranker else None
        self.qcache = QueryCache()

    # ---------- ingestion ----------
    def ingest_pdf(self, path: str, doc_name: str) -> IngestReport:
        rep = ingest_pdf(path, doc_name, self.s, self.embedder, self.store)
        self.store.save()
        self.emb_cache.save()
        return rep

    @property
    def ready(self) -> bool:
        return len(self.store.chunks) > 0

    # ---------- retrieval (search ONCE, cached) ----------
    def retrieve(self, query: str) -> RetrieveResult:
        iv = self.store.index_version()
        sig = self.s.signature()
        trace = QueryTrace(query=query, preset=self.preset_name, index_version=iv)

        cached = self.qcache.get(iv, sig, query)
        if cached is not None:
            candidates, contexts = cached
            trace.cache_hit = True
            trace.candidate_ids = [c["id"] for c in candidates]
            trace.reranked_ids = [c["id"] for c in contexts]
            trace.used_source_ids = [c["id"] for c in contexts]
            trace.grounded = len(contexts) > 0
            return RetrieveResult(query, candidates, contexts, trace, trace.grounded)

        with trace.stage("embed"):
            qvec = self.embedder.embed_one(query)
        with trace.stage("hybrid"):
            candidates = self.store.hybrid_search(query, qvec, self.s.num_candidates, self.s.rrf_k)
        trace.candidate_ids = [c["id"] for c in candidates]
        trace.candidate_scores = [c["rrf_score"] for c in candidates]

        if self.reranker and candidates:
            with trace.stage("rerank"):
                contexts = self.reranker.rerank(query, candidates, self.s.max_contexts)
            trace.reranked_ids = [c["id"] for c in contexts]
            trace.reranked_scores = [c.get("rerank_score") for c in contexts]
        else:
            contexts = candidates[: self.s.max_contexts]
            trace.reranked_ids = [c["id"] for c in contexts]

        trace.used_source_ids = [c["id"] for c in contexts]
        trace.grounded = len(contexts) > 0
        self.qcache.put(iv, sig, query, [candidates, contexts])
        return RetrieveResult(query, candidates, contexts, trace, trace.grounded)

    # ---------- generation (streaming) ----------
    def answer_stream(self, query: str, history: list[dict] | None = None):
        """
        Yields events:
          {"type":"token","text":...}
          {"type":"done","answer":str,"contexts":list,"trace":QueryTrace}
        """
        history = history or []
        rr = self.retrieve(query)
        trace = rr.trace

        if rr.grounded:
            context_block = build_context_block(rr.contexts)
            user_msg = f"Context:\n{context_block}\n\nQuestion: {query}"
            system = RAG_SYSTEM_PROMPT
        else:
            user_msg = query
            system = FALLBACK_SYSTEM_PROMPT

        messages = history + [{"role": "user", "content": user_msg}]

        answer = ""
        with trace.stage("generate"):
            for ev in stream_completion(self.s.llm, system, messages,
                                        self.s.temperature, self.s.max_tokens):
                if "text" in ev:
                    answer += ev["text"]
                    yield {"type": "token", "text": ev["text"]}
                elif "usage" in ev:
                    trace.prompt_tokens = ev["usage"]["prompt_tokens"]
                    trace.completion_tokens = ev["usage"]["completion_tokens"]

        trace.persist(self.data_dir)
        yield {"type": "done", "answer": answer,
               "contexts": rr.contexts if rr.grounded else [], "trace": trace}
