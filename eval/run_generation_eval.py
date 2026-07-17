"""Generation evaluation — LLM-as-judge.

Scores two things SEPARATELY (same reason Recall@5/Hit@1/MRR and refusal
recall/false-refusal-rate were kept separate — different failure modes, one
blended number would hide which one is actually happening):
    - Faithfulness: is every claim in the answer backed by the retrieved context?
    - Relevance: does the answer actually address the question asked?

Refusal accuracy is NOT retested here — that's eval/run_refusal_eval.py's job.
This script only judges answerable questions that actually got a real answer;
if a sampled question gets wrongly refused, it's skipped and logged, not
silently counted as a failure of faithfulness/relevance.

Known caveat, stated honestly (goes in the README too): this uses Gemini to
judge Gemini's own answers. Self-preference bias is real — a model grading its
own output tends to be a little generous. Mitigation: read a handful of the
printed judge verdicts yourself before trusting the aggregate score, same way
the golden set itself was hand-verified rather than trusted blindly.

Run (from repo root):  python -m eval.run_generation_eval
"""
import json
import os
import random
import time
from pathlib import Path

import google.genai as genai

from courselens.config import GEMINI_MODEL
from courselens.retrieval import retrieve
from courselens.generation.answer import generate
from courselens.generation.prompts import REFUSAL_PHRASE

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"
SAMPLE_SIZE = 15
SECONDS_BETWEEN_CALLS = 4.5  # same free-tier throttle as run_refusal_eval.py

_judge_client: genai.Client | None = None


def _get_judge_client() -> genai.Client:
    global _judge_client
    if _judge_client is None:
        _judge_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _judge_client


def build_judge_prompt(question: str, context: list[dict], answer_text: str) -> str:
    context_text = "\n\n".join(c["text"] for c in context)
    return f"""You are a strict evaluator checking a RAG system's output.

Question: {question}

Retrieved Context (this is ALL the information the system was allowed to use):
{context_text}

Generated Answer:
{answer_text}

Judge two things, using ONLY the retrieved context above — not your own outside knowledge:

1. FAITHFUL: Is every factual claim in the Generated Answer actually supported by the
   Retrieved Context? If the answer includes ANY detail not present in the context, this is NO.
2. RELEVANT: Does the Generated Answer actually address the Question asked (regardless of
   whether it's faithful)?

Respond with EXACTLY these two lines, nothing else, no explanation:
FAITHFUL: YES or NO
RELEVANT: YES or NO
"""


def judge(question: str, context: list[dict], answer_text: str) -> tuple[bool, bool]:
    prompt = build_judge_prompt(question, context, answer_text)
    response = _get_judge_client().models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text.upper()
    faithful = "FAITHFUL: YES" in text
    relevant = "RELEVANT: YES" in text
    return faithful, relevant


def main():
    all_questions = [json.loads(line) for line in GOLDEN_SET_PATH.read_text().strip().split("\n")]
    answerable = [q for q in all_questions if q["answerable"]]

    random.seed(42)  # fixed seed -> reproducible sample, not cherry-picked after the fact
    sample = random.sample(answerable, min(SAMPLE_SIZE, len(answerable)))

    faithful_count = 0
    relevant_count = 0
    judged = 0
    skipped = 0

    for q in sample:
        context = retrieve(q["question"], mode="hybrid")
        answer_text = generate(q["question"], context)
        time.sleep(SECONDS_BETWEEN_CALLS)

        if answer_text.strip() == REFUSAL_PHRASE:
            print(f"SKIPPED (wrongly refused, can't judge)  {q['question'][:55]}")
            skipped += 1
            continue

        faithful, relevant = judge(q["question"], context, answer_text)
        time.sleep(SECONDS_BETWEEN_CALLS)

        faithful_count += faithful
        relevant_count += relevant
        judged += 1
        print(f"FAITHFUL={str(faithful):<5} RELEVANT={str(relevant):<5} {q['question'][:55]}")

    if judged == 0:
        print("\nNothing was judged — every sampled question got refused. Investigate before trusting any score.")
        return

    faithfulness_score = faithful_count / judged
    relevance_score = relevant_count / judged

    print("\n--- Summary ---")
    print(f"Judged {judged}/{len(sample)} sampled questions ({skipped} skipped due to refusal)")
    print(f"Faithfulness: {faithful_count}/{judged} = {faithfulness_score:.3f}")
    print(f"Relevance:    {relevant_count}/{judged} = {relevance_score:.3f}")

    text = RESULTS_PATH.read_text()
    text = text.replace(
        "| Faithfulness (LLM-judge)   |       |",
        f"| Faithfulness (LLM-judge)   | {faithfulness_score:.3f} ({faithful_count}/{judged}) |",
    )
    text = text.replace(
        "| Answer relevance           |       |",
        f"| Answer relevance           | {relevance_score:.3f} ({relevant_count}/{judged}) |",
    )
    RESULTS_PATH.write_text(text)
    print(f"\nWrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
