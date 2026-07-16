"""
Synthetic golden-set builder (SCAFFOLD).

Speeds up authoring an eval set: it reads your indexed chunks, asks a strong LLM
to write a question + reference answer + the exact snippet that answers it, and
writes candidate items to eval/datasets/golden_synthetic.jsonl.

IMPORTANT: a human MUST review every item before trusting it. Unreviewed
synthetic evals mostly measure whether two models agree, not whether answers are
correct. This is a starting point, not a substitute for judgment.

Usage:
    python -m eval.build_golden --preset Balanced --n 20
"""
from __future__ import annotations
import argparse, json, os, random, re

from rag_app.config import PRESETS, JUDGE_LLM
from rag_app.pipeline import RagEngine

PROMPT = """You are writing a QA evaluation item from the passage below.
Return STRICT JSON with keys: question, ground_truth, source_snippet.
- question: a specific question answerable ONLY from this passage.
- ground_truth: the correct, concise answer.
- source_snippet: a short exact substring (3-8 words) copied verbatim from the passage that contains the answer.
Passage:
\"\"\"{passage}\"\"\""""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="Balanced")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    from litellm import completion
    engine = RagEngine(PRESETS[args.preset], preset_name=args.preset)
    if not engine.ready:
        raise SystemExit("No index found. Ingest PDFs via the app first.")

    chunks = [c for c in engine.store.chunks if len(c["text"].split()) > 40]
    random.shuffle(chunks)
    out_path = os.path.join("eval", "datasets", "golden_synthetic.jsonl")
    written = 0
    with open(out_path, "w") as f:
        for c in chunks[: args.n]:
            resp = completion(model=JUDGE_LLM, temperature=0.3,
                              messages=[{"role": "user", "content": PROMPT.format(passage=c["text"])}])
            raw = resp.choices[0].message.content
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                continue
            try:
                item = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            item["source_doc"] = c["doc_name"]
            # keep only items whose snippet truly appears in the passage
            if item.get("source_snippet", "").lower() in c["text"].lower():
                f.write(json.dumps(item) + "\n")
                written += 1
    print(f"Wrote {written} candidate items to {out_path}. REVIEW THEM before use.")


if __name__ == "__main__":
    main()
