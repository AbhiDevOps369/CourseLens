"""Reciprocal Rank Fusion — combine dense + sparse ranked lists.

RRF:  score(d) = sum over each retriever i of  1 / (RRF_K + rank_i(d))

Rank-based on purpose: BM25 scores and cosine similarities live on totally
different, incomparable scales, so we can't just add the raw numbers together.
Rank position ("this was #1", "this was #3") means the same thing regardless
of which retriever produced it — that's what makes fusion fair.

Input: dense_results and sparse_results are each a list[dict] already sorted
best-match-first (exactly what dense.retrieve() / sparse.retrieve() hand back),
each dict has at least an "id" key plus the usual video_id/start/end/text.

Output: one fused list[dict], re-ranked by combined RRF score, top `top_n`.
"""

RRF_K = 60  # standard constant from the original RRF paper


def fuse(dense_results: list[dict], sparse_results: list[dict], top_n: int = 10) -> list[dict]:
    scores = {}  # chunk id -> accumulated RRF score
    docs = {}    # chunk id -> the actual chunk dict, so we can look up its text/video_id/etc later

    for rank, doc in enumerate(dense_results, start=1):
        scores[doc["id"]]=scores.get(doc["id"], 0)+ 1/(RRF_K + rank)
        docs[doc["id"]] = doc
   
    for rank,doc in enumerate(sparse_results,start=1):
        scores[doc["id"]]=scores.get(doc["id"],0) + 1/(RRF_K + rank)
        docs[doc["id"]]=doc

    
    top_ids=sorted(scores.items(),key =lambda pair:pair[1],reverse=True)[:top_n]
    
    results=[]
    for chunk_id, fused_score in top_ids:
        results.append({**docs[chunk_id], "score": fused_score})

    return results


if __name__ == "__main__":
    from courselens.retrieval import dense, sparse

    question = input("Ask a question: ")
    d = dense.retrieve(question, k=20)
    s = sparse.retrieve(question, k=20)
    for i, c in enumerate(fuse(d, s, top_n=10), 1):
        print(f'{i}. [{c["id"]}] rrf={c["score"]:.4f}  {c["text"][:90].strip()}...')
