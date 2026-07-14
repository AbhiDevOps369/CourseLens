"""Reciprocal Rank Fusion — combine dense + sparse ranked lists.  [TO BUILD — Phase 2]

RRF:  score(d) = Σ_i  1 / (60 + rank_i(d))   over each retriever i.
Rank-based, so it needs no score calibration between BM25 and cosine (which live on
different, query-dependent scales). Take the fused top-10 into the reranker.

Also expose the single entry point the eval harness will ablate over:
    def retrieve(query, mode="dense"|"hybrid"|"hybrid+rerank") -> pd.DataFrame
"""
raise NotImplementedError("Build RRF fusion here — roadmap Phase 2.")
