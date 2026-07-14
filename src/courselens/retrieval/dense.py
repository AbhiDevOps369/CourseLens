"""Dense retrieval — embed the query, ANN search over the Chroma collection, top-k.

Phase 1 rewrite: the old joblib DataFrame + brute-force sklearn cosine is gone.
We now query the persistent ChromaDB 'transcript' collection built by
ingest/index.py. Chroma loads once (client cached) and uses its HNSW index, so
this scales past the point where scanning the whole matrix would.

Returns a uniform list[dict] shape shared by every retriever (sparse, fusion,
rerank) so the eval harness (Phase 3) can ablate over them interchangeably:
    {id, video_id, start, end, text, score}
"""
import numpy as np
import requests
import chromadb

from courselens.config import (
    CHROMA_DIR, TRANSCRIPT_COLLECTION, EMBEDDING_MODEL, OLLAMA_EMBED_URL, TOP_K,
)

_collection = None  # cached so repeated queries don't reopen the store


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(TRANSCRIPT_COLLECTION)
    return _collection


def embed_query(query: str) -> list[float]:
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBEDDING_MODEL, "input": [query]})
    r.raise_for_status()
    return r.json()["embeddings"][0]


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k most similar chunks by dense cosine similarity."""
    col = _get_collection()
    res = col.query(query_embeddings=[embed_query(query)], n_results=k)
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]  # cosine distance; similarity = 1 - distance
    out = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({
            "id": cid,
            "video_id": meta["video_id"],
            "start": meta["start"],
            "end": meta["end"],
            "text": doc,
            "score": 1.0 - float(dist),
        })
    return out


if __name__ == "__main__":
    question = input("Ask a question: ")
    for i, c in enumerate(retrieve(question), 1):
        print(f'{i}. [{c["id"]}] sim={c["score"]:.3f}  {c["text"][:90].strip()}...')