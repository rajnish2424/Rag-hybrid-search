"""
Embedder. Default backend is fastembed (BGE ONNX models) which runs on CPU
cheaply -- embedding is NOT the expensive part of RAG, generation is. A hosted
backend (via LiteLLM) is available if you want zero local model weights.

All vectors are L2-normalized so cosine similarity == dot product.
"""
from __future__ import annotations
import numpy as np
from .config import Settings
from .cache import EmbeddingCache


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)


class Embedder:
    def __init__(self, settings: Settings, cache: EmbeddingCache | None = None):
        self.s = settings
        self.cache = cache
        self._model = None  # lazy-loaded

    def _fastembed(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.s.embed_model)
        return self._model

    def _embed_raw(self, texts: list[str]) -> np.ndarray:
        if self.s.embed_backend == "fastembed":
            model = self._fastembed()
            vecs = np.array(list(model.embed(texts)), dtype=np.float32)
        else:  # litellm hosted embedder
            from litellm import embedding
            resp = embedding(model=self.s.embed_model, input=texts)
            vecs = np.array([d["embedding"] for d in resp["data"]], dtype=np.float32)
        return _l2(vecs)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, using the on-disk cache where possible."""
        if not texts:
            return np.zeros((0, self.s.embed_dim), dtype=np.float32)
        if self.cache is None:
            return self._embed_raw(texts)

        out: list[np.ndarray | None] = [None] * len(texts)
        missing_idx, missing_txt = [], []
        for i, t in enumerate(texts):
            hit = self.cache.get(t)
            if hit is None:
                missing_idx.append(i)
                missing_txt.append(t)
            else:
                out[i] = hit
        if missing_txt:
            fresh = self._embed_raw(missing_txt)
            for j, i in enumerate(missing_idx):
                out[i] = fresh[j]
                self.cache.put(missing_txt[j], fresh[j])
        return np.vstack(out).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
