"""
Caching. Two independent caches, both keyed by content so they're safe.

1) EmbeddingCache  - never embed identical text twice (persists to disk).
2) QueryCache      - repeated/identical questions skip the whole search+rerank.
                     Keyed by (index_version, settings_signature, normalized_query),
                     so it self-invalidates when documents or settings change.
"""
from __future__ import annotations
import os
import json
import hashlib
import pickle
import numpy as np


def _norm_query(q: str) -> str:
    return " ".join(q.lower().split())


def _key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


class EmbeddingCache:
    def __init__(self, data_dir: str, signature: str):
        self.path = os.path.join(data_dir, f"emb_cache_{signature}.pkl")
        self._d: dict[str, np.ndarray] = {}
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self._d = pickle.load(f)

    def get(self, text: str):
        return self._d.get(_key(text))

    def put(self, text: str, vec: np.ndarray):
        self._d[_key(text)] = vec

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self._d, f)


class QueryCache:
    """In-memory LRU-ish query cache (small; resets each process)."""
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._d: dict[str, list] = {}

    def _k(self, index_version: str, sig: str, query: str) -> str:
        return _key(index_version, sig, _norm_query(query))

    def get(self, index_version: str, sig: str, query: str):
        return self._d.get(self._k(index_version, sig, query))

    def put(self, index_version: str, sig: str, query: str, value: list):
        if len(self._d) >= self.capacity:
            self._d.pop(next(iter(self._d)))
        self._d[self._k(index_version, sig, query)] = value
