"""Single entry point the eval harness (Day 2) and the API (Day 3) will call.

    retrieve(query, mode="dense")           -> just dense.retrieve()
    retrieve(query, mode="hybrid")          -> dense + sparse, fused via RRF
    retrieve(query, mode="hybrid+rerank")   -> dense + sparse, fused, then cross-encoder reranked

The candidate pool is widened before fusion (CANDIDATE_K) and fusion keeps a
wider shortlist before reranking (FUSION_TOP_N) than the final `k` you actually
want back — each stage gets more to work with than it hands to the next one.
"""
from courselens.config import TOP_K
from courselens.retrieval import dense, sparse, fusion, rerank

CANDIDATE_K = 20    # how many each of dense/sparse fetch before fusion
FUSION_TOP_N = 10   # how many fusion keeps before reranking


def retrieve(query: str, mode: str = "dense", k: int = TOP_K) -> list[dict]:
    if mode == "dense":
        return dense.retrieve(query, k=k)

    d = dense.retrieve(query, k=CANDIDATE_K)
    s = sparse.retrieve(query, k=CANDIDATE_K)

    if mode == "hybrid":
        return fusion.fuse(d, s, top_n=k)

    if mode == "hybrid+rerank":
        fused = fusion.fuse(d, s, top_n=FUSION_TOP_N)
        return rerank.rerank(query, fused, k=k)

    raise ValueError(f"unknown mode: {mode!r} (expected 'dense', 'hybrid', or 'hybrid+rerank')")
