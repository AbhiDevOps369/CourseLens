"""Retrieval evaluation harness — the differentiator.

For each retriever mode (dense / hybrid / hybrid+rerank), run every
ANSWERABLE question in golden_set.jsonl and compute:
    Recall@5 : fraction of questions where a gold chunk appears in the top-5
    Hit@1    : fraction where the top result is a gold chunk
    MRR@10   : mean of 1/rank of the first gold chunk found in the top-10

The 7 unanswerable/off-topic questions in the golden set are deliberately
skipped here — they exist for Day 3's refusal-accuracy check, not retrieval
quality, since there's no "correct chunk" to measure them against.

Numbers are never invented — whatever this script prints is what gets written
to results.md, negative or positive.

Run (from the repo root):  python -m eval.run_retrieval_eval
"""
import json
from pathlib import Path

from courselens.retrieval import retrieve

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"
MODES = ["dense", "hybrid", "hybrid+rerank"]
MODE_LABELS = {
    "dense": "Dense (bge-m3)",
    "hybrid": "Hybrid (RRF)",
    "hybrid+rerank": "Hybrid + reranker",
}


def load_answerable_questions() -> list[dict]:
    lines = GOLDEN_SET_PATH.read_text().strip().split("\n")
    all_questions = [json.loads(line) for line in lines]
    return [q for q in all_questions if q["answerable"]]


def evaluate_mode(mode: str, questions: list[dict]) -> dict:
    recall_hits = 0
    hit1_hits = 0
    reciprocal_ranks = []

    for q in questions:
        results = retrieve(q["question"], mode=mode, k=10)
        result_ids = [r["id"] for r in results]
        gold_ids = set(q["expected_chunk_ids"])

        # Recall@5: does ANY gold id show up anywhere in the top 5?
        if any(rid in gold_ids for rid in result_ids[:5]):
            recall_hits += 1

        # Hit@1: is the very first result a gold id?
        if result_ids and result_ids[0] in gold_ids:
            hit1_hits += 1

        # MRR@10: reciprocal rank of the FIRST gold id found in the top 10
        # (0 if no gold id appears anywhere in the top 10 at all)
        rr = 0.0
        for rank, rid in enumerate(result_ids[:10], start=1):
            if rid in gold_ids:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

    n = len(questions)
    return {
        "recall@5": recall_hits / n,
        "hit@1": hit1_hits / n,
        "mrr@10": sum(reciprocal_ranks) / n,
    }


def write_results_table(all_results: dict) -> None:
    lines = [
        "# Evaluation results",
        "",
        "> Filled with numbers from an actually-run eval (eval/run_retrieval_eval.py).",
        "",
        "## Retrieval ablation (Phase 3)",
        "",
        "| Retriever          | Recall@5 | Hit@1 | MRR@10 |",
        "|--------------------|----------|-------|--------|",
    ]
    for mode in MODES:
        m = all_results[mode]
        lines.append(
            f"| {MODE_LABELS[mode]:<18} | {m['recall@5']:.3f}    | {m['hit@1']:.3f} | {m['mrr@10']:.3f}  |"
        )
    lines += [
        "",
        "## Generation quality (Phase 4)",
        "",
        "| Metric                     | Score |",
        "|----------------------------|-------|",
        "| Faithfulness (LLM-judge)   |       |",
        "| Answer relevance           |       |",
        "| Refusal accuracy (OOD)     |       |",
    ]
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


def main():
    questions = load_answerable_questions()
    print(f"Evaluating on {len(questions)} answerable questions (of {len(questions)} loaded)...")

    all_results = {}
    for mode in MODES:
        print(f"\nRunning mode={mode} ...")
        metrics = evaluate_mode(mode, questions)
        all_results[mode] = metrics
        print(f"  Recall@5={metrics['recall@5']:.3f}  Hit@1={metrics['hit@1']:.3f}  MRR@10={metrics['mrr@10']:.3f}")

    write_results_table(all_results)
    print(f"\nWrote ablation table to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
