"""Manual sanity check — run one question through all 3 retrieval modes side by side.

    python -m courselens.retrieval
"""
from courselens.retrieval import retrieve

if __name__ == "__main__":
    question = input("Ask a question: ")
    for mode in ("dense", "hybrid", "hybrid+rerank"):
        print(f"\n--- mode={mode} ---")
        for i, c in enumerate(retrieve(question, mode=mode, k=5), 1):
            print(f'{i}. [{c["id"]}] score={c["score"]:.4f}  {c["text"][:80].strip()}...')
