"""Sparse (keyword) retrieval — BM25 over chunk text.  [TO BUILD — roadmap Phase 2]

Why: dense embeddings miss exact lexical matches (API names like `useEffect`,
`Appwrite`, error strings). BM25 catches those. You'll build the index in
ingest/index.py (rank_bm25.BM25Okapi over tokenized chunk texts) and query it here.

Target signature to match dense.retrieve():
    def retrieve(query: str, k: int = TOP_K) -> pd.DataFrame: ...
"""
raise NotImplementedError("Build BM25 retrieval here — roadmap Phase 2.")
