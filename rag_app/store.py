"""
Storage + hybrid retrieval.

A deliberately simple, transparent store so the app runs ANYWHERE with no server:
- Dense vectors in a NumPy matrix (cosine == dot product, vectors are normalized).
- Keyword index via BM25 (rank_bm25, pure Python).
- Fusion via Reciprocal Rank Fusion (RRF): the standard way to blend two ranked
  lists without having to calibrate their raw scores against each other.

For production scale you'd swap this class for Qdrant / LanceDB / pgvector, but
the interface (add / hybrid_search) would stay the same.
"""
from __future__ import annotations
import os
import json
import hashlib
import numpy as np


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class Store:
    def __init__(self, data_dir: str, embed_dim: int, embedder_signature: str):
        self.data_dir = data_dir
        self.embed_dim = embed_dim
        self.embedder_signature = embedder_signature
        self.chunks: list[dict] = []          # {id, doc_name, page, text, chash}
        self.vectors = np.zeros((0, embed_dim), dtype=np.float32)
        self._seen_hashes: set[str] = set()
        self._bm25 = None
        self._bm25_tokens: list[list[str]] = []

    # ---------- persistence ----------
    @property
    def _meta_path(self):
        return os.path.join(self.data_dir, "index_meta.json")

    @property
    def _vec_path(self):
        return os.path.join(self.data_dir, "index_vectors.npy")

    def load(self) -> bool:
        if not (os.path.exists(self._meta_path) and os.path.exists(self._vec_path)):
            return False
        meta = json.load(open(self._meta_path))
        if meta.get("embedder_signature") != self.embedder_signature:
            # Different embedding space -> old index is unusable.
            return False
        self.chunks = meta["chunks"]
        self._seen_hashes = {c["chash"] for c in self.chunks}
        self.vectors = np.load(self._vec_path)
        self._build_bm25()
        return True

    def save(self):
        os.makedirs(self.data_dir, exist_ok=True)
        json.dump(
            {"embedder_signature": self.embedder_signature, "chunks": self.chunks},
            open(self._meta_path, "w"),
        )
        np.save(self._vec_path, self.vectors)

    # ---------- index version (for cache invalidation) ----------
    def index_version(self) -> str:
        h = hashlib.sha256()
        h.update(self.embedder_signature.encode())
        for c in self.chunks:
            h.update(c["chash"].encode())
        return h.hexdigest()[:12]

    @property
    def num_docs(self) -> int:
        return len({c["doc_name"] for c in self.chunks})

    # ---------- ingestion ----------
    def add(self, new_chunks: list[dict], vectors: np.ndarray) -> int:
        """new_chunks: [{doc_name, page, text}]. De-dupes on content hash."""
        keep_chunks, keep_vecs = [], []
        for ch, v in zip(new_chunks, vectors):
            chash = content_hash(ch["text"])
            if chash in self._seen_hashes:
                continue
            self._seen_hashes.add(chash)
            ch = {**ch, "chash": chash, "id": f"c{len(self.chunks) + len(keep_chunks)}"}
            keep_chunks.append(ch)
            keep_vecs.append(v)
        if keep_chunks:
            self.chunks.extend(keep_chunks)
            self.vectors = np.vstack([self.vectors, np.array(keep_vecs, dtype=np.float32)])
            self._build_bm25()
        return len(keep_chunks)

    # ---------- search ----------
    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        self._bm25_tokens = [c["text"].lower().split() for c in self.chunks]
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

    def _dense_rank(self, qvec: np.ndarray, k: int) -> list[int]:
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ qvec  # cosine (normalized)
        return list(np.argsort(-scores)[:k])

    def _sparse_rank(self, query: str, k: int) -> list[int]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        return list(np.argsort(-scores)[:k])

    def hybrid_search(self, query: str, qvec: np.ndarray, k: int, rrf_k: int = 60) -> list[dict]:
        """Return up to k fused candidates: [{**chunk, rrf_score}] best first."""
        dense = self._dense_rank(qvec, k)
        sparse = self._sparse_rank(query, k)
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, idx in enumerate(sparse):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        ordered = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{**self.chunks[i], "rrf_score": round(s, 5)} for i, s in ordered]
