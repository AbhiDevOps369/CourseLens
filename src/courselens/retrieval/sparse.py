"""Sparse (keyword) retrieval — BM25 over chunk text.

Loads the BM25 index + metadata pickled by ingest/index.py at BM25_PATH.
Must reuse the exact same tokenize() used at build time in ingest/index.py —
if you tokenize differently here, your query terms won't match the vocabulary
the index was built on, and every score will be garbage.

Target shape — matches dense.retrieve() exactly, so fusion.py can treat both
retrievers interchangeably:
    list[dict] with keys: id, video_id, start, end, text, score
"""
import pickle

import numpy as np

from courselens.config import BM25_PATH, TOP_K
from courselens.ingest.index import tokenize  # reuse the SAME tokenizer as build time

_bm25 = None
_meta = None


def _load():
    global _bm25, _meta
    if _bm25 is None:
        with open(BM25_PATH,'rb') as file:
            data=pickle.load(file)
            _bm25= data["bm25"]
            _meta=data["meta"]
        pass
    return _bm25, _meta


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    bm25, meta = _load()

    tokenized_query=tokenize(query)

    scores=bm25.get_scores(tokenized_query)

    ranked_scores=np.argsort(scores)[-k:][::-1]
    results=[]

    for i in ranked_scores:
        results.append({
            "id": meta[i]["id"], "video_id": meta[i]["video_id"],
           "start": meta[i]["start"], "end": meta[i]["end"],
           "text": meta[i]["text"], "score": float(scores[i]),
        })

    return results

  


if __name__ == "__main__":
    question = input("Ask a question: ")
    for i, c in enumerate(retrieve(question), 1):
        print(f'{i}. [{c["id"]}] bm25={c["score"]:.3f}  {c["text"][:90].strip()}...')
