"""
Central configuration.

The big idea for a PM: almost every "quality vs speed vs cost" lever in a RAG
system is a knob here. Instead of asking a non-technical user to paste model
paths, we expose three PRESETS. Each preset is just a bundle of these knobs.

Heavy LLM generation runs on a HOSTED provider (Groq by default) so a weak CPU
never has to run a large model. Embeddings + reranking are tiny ONNX models that
run locally on CPU cheaply.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Literal
import hashlib
import json


@dataclass
class Settings:
    # ---- Generation (hosted) ----
    # LiteLLM model string. "groq/<model>" uses Groq. You could also use
    # "openai/gpt-4o-mini", "gemini/gemini-1.5-flash", etc. (set the matching API key).
    llm: str = "groq/llama-3.1-8b-instant"
    temperature: float = 0.2
    max_tokens: int = 1024

    # ---- Embeddings (local ONNX via fastembed) ----
    embed_backend: Literal["fastembed", "litellm"] = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384  # bge-small=384, bge-base=768, bge-large=1024

    # ---- Reranker (local ONNX via FlashRank) ----
    use_reranker: bool = True
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"

    # ---- Chunking ----
    chunk_size_words: int = 220   # target chunk size, in words (~ tokens*0.75)
    chunk_overlap_words: int = 40  # overlap between consecutive chunks

    # ---- Retrieval ----
    num_candidates: int = 10   # how many the hybrid search fetches (wide, cheap)
    max_contexts: int = 5      # how many survive rerank and go to the LLM (narrow, precise)
    rrf_k: int = 60            # Reciprocal Rank Fusion constant (standard default)

    def embedder_signature(self) -> str:
        """Identity of the embedding space. If this changes, the index is stale."""
        return f"{self.embed_backend}:{self.embed_model}:{self.embed_dim}"

    def signature(self) -> str:
        """Stable hash of settings that affect retrieval results (for caching)."""
        keys = ["embedder_signature", "chunk_size_words", "chunk_overlap_words",
                "num_candidates", "max_contexts", "use_reranker", "reranker_model", "rrf_k"]
        payload = {"embedder_signature": self.embedder_signature()}
        payload.update({k: getattr(self, k) for k in keys if k != "embedder_signature"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)


# Three presets. "Balanced" is the default a non-developer should pick.
PRESETS: dict[str, Settings] = {
    "Fast": Settings(
        llm="groq/llama-3.1-8b-instant",
        embed_model="BAAI/bge-small-en-v1.5", embed_dim=384,
        use_reranker=False,                 # skip rerank to save latency
        chunk_size_words=180, chunk_overlap_words=30,
        num_candidates=8, max_contexts=4,
        reranker_model="ms-marco-MiniLM-L-6-v2",
    ),
    "Balanced": Settings(
        llm="groq/llama-3.1-8b-instant",
        embed_model="BAAI/bge-base-en-v1.5", embed_dim=768,
        use_reranker=True,
        chunk_size_words=220, chunk_overlap_words=40,
        num_candidates=10, max_contexts=5,
        reranker_model="ms-marco-MiniLM-L-12-v2",
    ),
    "High-quality": Settings(
        llm="groq/llama-3.3-70b-versatile",  # bigger model, still free on Groq
        embed_model="BAAI/bge-large-en-v1.5", embed_dim=1024,
        use_reranker=True,
        chunk_size_words=240, chunk_overlap_words=60,
        num_candidates=20, max_contexts=8,
        reranker_model="ms-marco-MiniLM-L-12-v2",
    ),
}

DEFAULT_PRESET = "Balanced"

# A stronger model is used as the EVALUATION JUDGE than as the product LLM.
# Lesson: your judge should be at least as capable as the model it grades.
JUDGE_LLM = "groq/llama-3.3-70b-versatile"

# Where the on-disk index + caches live.
DATA_DIR = "storage"
