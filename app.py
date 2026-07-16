"""
Streamlit UI.

Non-developer onboarding: pick a PRESET (Fast / Balanced / High-quality) instead
of pasting model paths and connection strings. Every answer shows its CITATIONS
(numbered sources) and an expandable TRACE (what was retrieved/reranked, latency
per stage, tokens). All heavy generation happens on a hosted API.
"""
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag_app.config import PRESETS, DEFAULT_PRESET
from rag_app.pipeline import RagEngine

load_dotenv()

st.set_page_config(page_title="Hybrid-Search RAG (Pro)", layout="wide")


def get_engine(preset_name: str) -> RagEngine:
    key = f"engine::{preset_name}"
    if key not in st.session_state:
        st.session_state[key] = RagEngine(PRESETS[preset_name], preset_name=preset_name)
    return st.session_state[key]


with st.sidebar:
    st.title("⚙️ Setup")

    if not os.getenv("GROQ_API_KEY"):
        st.warning("Set GROQ_API_KEY (free at console.groq.com) in your environment or .env file.")

    preset_name = st.radio(
        "Quality preset",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(DEFAULT_PRESET),
        help="Fast = lowest latency. Balanced = good default. High-quality = biggest model + best retrieval.",
    )
    s = PRESETS[preset_name]
    st.caption(
        f"**LLM:** {s.llm}  \n**Embedder:** {s.embed_model} ({s.embed_dim}d)  \n"
        f"**Reranker:** {'on — ' + s.reranker_model if s.use_reranker else 'off'}  \n"
        f"**Chunks:** {s.chunk_size_words}w / {s.chunk_overlap_words}w overlap  \n"
        f"**Retrieve→context:** {s.num_candidates} → {s.max_contexts}"
    )

engine = get_engine(preset_name)
st.session_state.setdefault("chat", [])

st.title("🔎 Hybrid-Search RAG — with citations, caching & tracing")

# --- Ingestion ---
uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
if uploaded:
    for uf in uploaded:
        with st.spinner(f"Indexing {uf.name}…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name
            try:
                rep = engine.ingest_pdf(tmp_path, uf.name)
            finally:
                os.remove(tmp_path)
        msg = f"**{uf.name}**: {rep.pages} pages, {rep.chunks_added} chunks added"
        if rep.chunks_skipped_duplicate:
            msg += f", {rep.chunks_skipped_duplicate} duplicates skipped"
        st.success(msg)
        if rep.warning:
            st.warning(rep.warning)

if engine.ready:
    st.caption(f"Index: {engine.store.num_docs} document(s), {len(engine.store.chunks)} chunks "
               f"· version `{engine.store.index_version()}`")

# --- Chat history ---
for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        if turn.get("contexts"):
            with st.expander(f"📚 Sources ({len(turn['contexts'])})"):
                for i, c in enumerate(turn["contexts"], 1):
                    st.markdown(f"**[{i}] {c['doc_name']} — p.{c['page']}**")
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
        if turn.get("trace"):
            t = turn["trace"]
            with st.expander("🔬 Trace"):
                st.json({
                    "cache_hit": t.cache_hit, "grounded": t.grounded,
                    "latency_ms": t.stage_ms, "total_ms": t.total_ms,
                    "tokens": {"prompt": t.prompt_tokens, "completion": t.completion_tokens},
                    "candidates": t.candidate_ids, "used_sources": t.used_source_ids,
                })

# --- New question ---
if not engine.ready:
    st.info("Upload a PDF to begin.")
else:
    q = st.chat_input("Ask a question about your documents…")
    if q:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            box = st.empty()
            answer, contexts, trace = "", [], None
            hist = [m for turn in st.session_state.chat
                    for m in ({"role": "user", "content": turn["q"]},
                              {"role": "assistant", "content": turn["a"]})]
            for ev in engine.answer_stream(q, history=hist):
                if ev["type"] == "token":
                    answer += ev["text"]
                    box.markdown(answer + "▌")
                else:
                    contexts, trace = ev["contexts"], ev["trace"]
            box.markdown(answer)
            if contexts:
                with st.expander(f"📚 Sources ({len(contexts)})"):
                    for i, c in enumerate(contexts, 1):
                        st.markdown(f"**[{i}] {c['doc_name']} — p.{c['page']}**")
                        st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
        st.session_state.chat.append({"q": q, "a": answer, "contexts": contexts, "trace": trace})
        st.rerun()
