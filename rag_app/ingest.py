"""
Ingestion. Extracts text per page (so citations can reference page numbers),
chunks it, embeds it, and adds it to the store (which de-dupes).

Robustness note: scanned/image PDFs have no extractable text. We DETECT that
(near-empty extraction) and warn, rather than silently indexing nothing. Real
OCR (e.g. pytesseract/ocrmypdf) is the production fix; we flag it here.
"""
from __future__ import annotations
from dataclasses import dataclass
from .chunking import chunk_text
from .config import Settings
from .embeddings import Embedder
from .store import Store


@dataclass
class IngestReport:
    doc_name: str
    pages: int
    chunks_added: int
    chunks_skipped_duplicate: int
    warning: str | None = None


def extract_pages(file_path: str) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    return [(page.extract_text() or "") for page in reader.pages]


def ingest_pdf(file_path: str, doc_name: str, settings: Settings,
               embedder: Embedder, store: Store) -> IngestReport:
    pages = extract_pages(file_path)
    total_chars = sum(len(p.strip()) for p in pages)

    warning = None
    if total_chars < 20 * max(1, len(pages)):  # almost no text per page
        warning = ("Very little text extracted \u2014 this may be a scanned/image PDF. "
                   "OCR is required to index it (not enabled in this build).")

    new_chunks = []
    for pageno, text in enumerate(pages, start=1):
        for ch in chunk_text(text, settings.chunk_size_words, settings.chunk_overlap_words):
            new_chunks.append({"doc_name": doc_name, "page": pageno, "text": ch})

    before = len(store.chunks)
    if new_chunks:
        vecs = embedder.embed([c["text"] for c in new_chunks])
        added = store.add(new_chunks, vecs)
    else:
        added = 0
    skipped = len(new_chunks) - added

    return IngestReport(
        doc_name=doc_name, pages=len(pages), chunks_added=added,
        chunks_skipped_duplicate=max(0, skipped), warning=warning,
    )
