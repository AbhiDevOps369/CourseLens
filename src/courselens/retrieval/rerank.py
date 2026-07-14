"""Cross-encoder reranking — BAAI/bge-reranker-base.  [TO BUILD — roadmap Phase 2]

Retrieve-many-cheap (bi-encoder + BM25), rerank-few-expensive: score the fused top-10
(query, chunk) pairs with a cross-encoder and keep the final top-5. CPU is fine for
~10 pairs. Use sentence-transformers CrossEncoder.
"""
raise NotImplementedError("Build cross-encoder reranking here — roadmap Phase 2.")
