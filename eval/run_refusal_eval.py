"""Refusal accuracy eval — does the system correctly refuse unanswerable
questions, and correctly NOT refuse answerable ones?

Only tests `hybrid` mode — that's the one that ships (see Day 2 finding:
hybrid+rerank regressed on Hit@1 and was ~5x slower). No reason to spend
extra Gemini calls validating refusal on modes nobody will use.

Reports TWO numbers, not one blended score, same reason Recall@5/Hit@1/MRR
were kept separate — refusing a good question and answering a bad one are
two different failure modes, and a single blended accuracy number would hide
which one is actually happening:
    - refusal recall: of the unanswerable questions, what fraction got refused
    - false-refusal rate: of the answerable questions, what fraction got
      WRONGLY refused (a real cost — users blocked from real answers)

Run (from repo root):  python -m eval.run_refusal_eval
"""
import json
import time
from pathlib import Path

from courselens.generation.answer import answer
from courselens.generation.prompts import REFUSAL_PHRASE

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Free-tier Gemini quota is 15 requests/minute for this model. 4.5s between
# calls keeps us under that (~13/min) with some margin, instead of bursting
# 17 calls instantly and hitting a 429 RESOURCE_EXHAUSTED mid-run.
SECONDS_BETWEEN_CALLS = 4.5


def main():
    questions = [json.loads(line) for line in GOLDEN_SET_PATH.read_text().strip().split("\n")]

    unanswerable = [q for q in questions if not q["answerable"]]
    answerable = [q for q in questions if q["answerable"]]

    correct_refusals = 0
    print(f"--- Unanswerable questions (n={len(unanswerable)}) — should be refused ---")
    for q in unanswerable:
        result = answer(q["question"], mode="hybrid")
        refused = result.strip() == REFUSAL_PHRASE
        correct_refusals += refused
        print(f"{'REFUSED (correct)' if refused else 'ANSWERED (WRONG)':<20} {q['question'][:55]}")
        time.sleep(SECONDS_BETWEEN_CALLS)

    false_refusals = 0
    print(f"\n--- Answerable questions (n={len(answerable)}) — should NOT be refused ---")
    for q in answerable:
        result = answer(q["question"], mode="hybrid")
        wrongly_refused = result.strip() == REFUSAL_PHRASE
        false_refusals += wrongly_refused
        print(f"{'REFUSED (WRONG)' if wrongly_refused else 'answered (correct)':<20} {q['question'][:55]}")
        time.sleep(SECONDS_BETWEEN_CALLS)

    refusal_recall = correct_refusals / len(unanswerable)
    false_refusal_rate = false_refusals / len(answerable)

    print("\n--- Summary ---")
    print(f"Refusal recall (caught the bad ones):     {correct_refusals}/{len(unanswerable)} = {refusal_recall:.3f}")
    print(f"False-refusal rate (blocked good ones):   {false_refusals}/{len(answerable)} = {false_refusal_rate:.3f}")

    # Fill in the existing "Refusal accuracy (OOD)" row in results.md
    text = RESULTS_PATH.read_text()
    old_line = "| Refusal accuracy (OOD)     |       |"
    new_line = (
        f"| Refusal accuracy (OOD)     | recall={refusal_recall:.2f} "
        f"({correct_refusals}/{len(unanswerable)}), false-refusals={false_refusal_rate:.2f} "
        f"({false_refusals}/{len(answerable)}) |"
    )
    if old_line in text:
        RESULTS_PATH.write_text(text.replace(old_line, new_line))
        print(f"\nWrote refusal results to {RESULTS_PATH}")
    else:
        print(f"\nCouldn't find the exact row to replace in {RESULTS_PATH} — update it by hand with the numbers above.")


if __name__ == "__main__":
    main()
