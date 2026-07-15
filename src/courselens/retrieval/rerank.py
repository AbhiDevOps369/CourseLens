"""Cross-encoder reranking — BAAI/bge-reranker-base.

Bi-encoder (dense.py) and BM25 (sparse.py) each encode the query and every
candidate SEPARATELY, which is exactly what makes them fast enough to search
a whole corpus, but it also means query and document never actually "look at"
each other while being scored.

A cross-encoder fixes that, at a cost: it takes the query and ONE candidate
document TOGETHER as a single input, so every query token can attend to every
document token. Much more accurate, but a full model forward pass per pair —
only affordable over a short shortlist, never the whole corpus. Hence the
pattern: retrieve many, cheaply (dense + sparse + RRF), then rerank few,
expensively (this file), right before handing the final top-k to Gemini.
"""
from sentence_transformers import CrossEncoder

from courselens.config import TOP_K

RERANKER_MODEL = "BAAI/bge-reranker-base"

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, candidates: list[dict], k: int = TOP_K) -> list[dict]:
    """Re-score fused candidates with a cross-encoder, return the new top-k.

    `candidates` is whatever fusion.fuse() handed back — already a short
    list (roughly 10), which is what makes this affordable to run at all.
    """
    if not candidates:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in candidates]
    ce_scores = model.predict(pairs)  # one relevance score per (query, doc) pair

    rescored = [{**c, "score": float(s)} for c, s in zip(candidates, ce_scores)]
    rescored.sort(key=lambda c: c["score"], reverse=True)
    return rescored[:k]


if __name__ == "__main__":
    from courselens.retrieval import dense, sparse, fusion

    question = input("Ask a question: ")
    d = dense.retrieve(question, k=20)
    s = sparse.retrieve(question, k=20)
    fused = fusion.fuse(d, s, top_n=10)
    for i, c in enumerate(rerank(question, fused, k=5), 1):
        print(f'{i}. [{c["id"]}] ce={c["score"]:.4f}  {c["text"][:90].strip()}...')
