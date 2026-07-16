"""
Chunking. This is the highest-leverage RAG knob, so it's explicit and tunable.

Strategy: split text into sentences (lightweight regex, no heavy spaCy download),
then greedily pack sentences into chunks up to `chunk_size_words`, carrying a
`chunk_overlap_words` tail into the next chunk. Overlap prevents an idea that
straddles a boundary from being lost. Splitting on sentences (not blind N-word
windows) keeps chunks semantically coherent.
"""
from __future__ import annotations
import re

# Simple sentence splitter: break after ., !, ? followed by whitespace + capital/quote/digit.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, size_words: int, overlap_words: int) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0

    def flush():
        nonlocal cur, cur_words
        if cur:
            chunks.append(" ".join(cur).strip())

    for sent in sentences:
        w = len(sent.split())
        if cur_words + w > size_words and cur:
            flush()
            # build overlap tail from the end of the previous chunk
            tail, tail_words = [], 0
            for s in reversed(cur):
                sw = len(s.split())
                if tail_words + sw > overlap_words and tail:
                    break
                tail.insert(0, s)
                tail_words += sw
                if tail_words >= overlap_words:
                    break
            # Guarantee at least one carried sentence when overlap is requested.
            if not tail and overlap_words > 0:
                tail = [cur[-1]]
            cur = tail[:]
            cur_words = tail_words
        cur.append(sent)
        cur_words += w
    flush()
    return [c for c in chunks if c]
