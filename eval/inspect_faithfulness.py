"""Permanent tool, not a scratch file — kept in eval/ alongside the other eval
scripts. Prints the raw retrieved context and generated answer for a chosen
list of questions, so a FAITHFUL=False (or any other suspicious) verdict from
run_generation_eval.py can actually be read and checked by hand instead of
trusted blindly. Edit QUESTIONS_TO_CHECK below to inspect different questions.

Run (from repo root):  python -m eval.inspect_faithfulness
"""
from courselens.retrieval import retrieve
from courselens.generation.answer import generate

QUESTIONS_TO_CHECK = [
    "Walk through setting up the auth Redux store: how is the store created, and what happens to state when a user logs in?",
    "Can you use multiple useEffect hooks inside a single component in this project?",
]


def main():
    for question in QUESTIONS_TO_CHECK:
        context = retrieve(question, mode="hybrid")
        answer_text = generate(question, context)

        print("=" * 80)
        print(f"QUESTION: {question}")
        print("-" * 80)
        print("RETRIEVED CONTEXT:")
        for c in context:
            print(f"\n[{c['id']}] (video {c['video_id']}, {c['start']}-{c['end']}s):")
            print(c["text"])
        print("-" * 80)
        print("GENERATED ANSWER:")
        print(answer_text)
        print()


if __name__ == "__main__":
    main()
