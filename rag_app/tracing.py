"""
Structured tracing / observability.

Every query records: what was retrieved, what survived reranking, latency per
stage, and token usage. This is how you debug ("why did it answer that?") and
cost-track. Traces are written as JSON lines to storage/traces.jsonl and also
kept in memory for the UI.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import time
import json
import os
import logging

logger = logging.getLogger("rag.trace")


@dataclass
class QueryTrace:
    query: str
    preset: str
    index_version: str
    cache_hit: bool = False
    grounded: bool = True            # False when we fell back to general knowledge
    stage_ms: dict = field(default_factory=dict)   # {"embed": .., "dense": .., ...}
    candidate_ids: list = field(default_factory=list)
    candidate_scores: list = field(default_factory=list)
    reranked_ids: list = field(default_factory=list)
    reranked_scores: list = field(default_factory=list)
    used_source_ids: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ts: float = field(default_factory=time.time)

    @property
    def total_ms(self) -> float:
        return round(sum(self.stage_ms.values()), 1)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage: `with trace.stage("dense"): ...`"""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stage_ms[name] = round((time.perf_counter() - t0) * 1000, 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_ms"] = self.total_ms
        d["total_tokens"] = self.total_tokens
        return d

    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "traces.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(self.to_dict()) + "\n")
        logger.info("query trace: %s ms, %s tokens, cache_hit=%s, grounded=%s",
                    self.total_ms, self.total_tokens, self.cache_hit, self.grounded)
