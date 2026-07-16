"""
Evaluation runner. This is the thing you actually run.

Two stages, measured separately (because they fail separately):

  RETRIEVAL  (did we fetch the right chunk?)  -> Recall@k, MRR   [computed here, no framework]
  GENERATION (was the answer right & grounded?) -> Ragas: faithfulness, answer
             relevancy, context precision/recall              [LLM-as-judge]

It also runs an ABLATION (reranker on vs off) so you can see the reranker's lift
as a number, not an opinion.

Run:
  # retrieval metrics only (fast, no LLM judge needed):
  python -m eval.run_eval --preset Balanced --no-generation

  # full eval incl. Ragas generation metrics (needs GROQ_API_KEY):
  python -m eval.run_eval --preset Balanced

  # ablation: compare reranker on vs off:
  python -m eval.run_eval --preset Balanced --ablate-rerank
"""
from __future__ import annotations
import argparse, json, os, copy

from rag_app.config import PRESETS, JUDGE_LLM
from rag_app.pipeline import RagEngine
from eval.retrieval_metrics import recall_at_k, reciprocal_rank, aggregate


def load_golden(path: str) -> list[dict]:
    items = []
    for fname in ("golden.jsonl", "golden_synthetic.jsonl"):
        fp = os.path.join(path, fname)
        if os.path.exists(fp):
            with open(fp) as f:
                items += [json.loads(l) for l in f if l.strip()]
    if not items:
        raise SystemExit(f"No golden data found in {path}")
    return items


def collect_answer(engine: RagEngine, q: str) -> tuple[str, list[dict]]:
    answer, contexts = "", []
    for ev in engine.answer_stream(q):
        if ev["type"] == "done":
            answer, contexts = ev["answer"], ev["contexts"]
    return answer, contexts


def run_retrieval(engine: RagEngine, golden: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    for item in golden:
        rr = engine.retrieve(item["question"])
        doc, snip = item["source_doc"], item["source_snippet"]
        row = {
            "question": item["question"],
            "recall@10_candidates": recall_at_k(rr.candidates, doc, snip, 10),
            "recall@5_candidates": recall_at_k(rr.candidates, doc, snip, 5),
            "mrr_candidates": reciprocal_rank(rr.candidates, doc, snip),
            "recall@5_final": recall_at_k(rr.contexts, doc, snip, 5),
            "mrr_final": reciprocal_rank(rr.contexts, doc, snip),
        }
        rows.append(row)
    return rows, aggregate(rows)


def run_generation(engine: RagEngine, golden: list[dict]) -> dict:
    """Ragas generation metrics. Returns {} with a note if Ragas isn't set up."""
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_groq import ChatGroq
    except Exception as e:  # pragma: no cover
        print(f"[generation] Ragas/langchain-groq not available ({e}). "
              f"Install eval deps or run with --no-generation.")
        return {}

    # A LangChain-compatible embeddings adapter around our local embedder.
    class _EmbAdapter:
        def __init__(self, embedder): self.e = embedder
        def embed_documents(self, texts): return [v.tolist() for v in self.e.embed(list(texts))]
        def embed_query(self, text): return self.e.embed_one(text).tolist()

    judge = LangchainLLMWrapper(ChatGroq(model=JUDGE_LLM.split("/", 1)[-1], temperature=0))
    embeddings = LangchainEmbeddingsWrapper(_EmbAdapter(engine.embedder))

    samples = []
    for item in golden:
        answer, contexts = collect_answer(engine, item["question"])
        samples.append(SingleTurnSample(
            user_input=item["question"],
            response=answer,
            retrieved_contexts=[c["text"] for c in contexts] or ["(no context retrieved)"],
            reference=item["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge, embeddings=embeddings,
    )
    df = result.to_pandas()
    numeric = df.select_dtypes("number")
    return {c: round(float(numeric[c].mean()), 4) for c in numeric.columns}


def evaluate_preset(preset_name: str, do_generation: bool) -> dict:
    engine = RagEngine(PRESETS[preset_name], preset_name=preset_name)
    if not engine.ready:
        raise SystemExit("No index found. Upload PDFs in the app (matching the golden set) first.")
    golden = load_golden(os.path.join("eval", "datasets"))
    print(f"\n=== Preset: {preset_name}  ({len(golden)} questions) ===")

    _, retr = run_retrieval(engine, golden)
    print("RETRIEVAL:")
    for k, v in retr.items():
        print(f"  {k:24s} {v}")

    gen = {}
    if do_generation:
        print("GENERATION (Ragas, LLM-judge)…")
        gen = run_generation(engine, golden)
        for k, v in gen.items():
            print(f"  {k:24s} {v}")

    return {"preset": preset_name, "retrieval": retr, "generation": gen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Balanced", choices=list(PRESETS.keys()))
    ap.add_argument("--no-generation", action="store_true", help="skip Ragas (retrieval only)")
    ap.add_argument("--ablate-rerank", action="store_true", help="compare reranker on vs off")
    args = ap.parse_args()

    results = []
    if args.ablate_rerank:
        base = PRESETS[args.preset]
        for flag in (True, False):
            variant = copy.deepcopy(base)
            variant.use_reranker = flag
            name = f"{args.preset}[rerank={'on' if flag else 'off'}]"
            PRESETS[name] = variant
            results.append(evaluate_preset(name, not args.no_generation))
    else:
        results.append(evaluate_preset(args.preset, not args.no_generation))

    out = os.path.join("eval", "results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
