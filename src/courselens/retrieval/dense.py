"""Dense retrieval — embed the query, cosine-similarity vs all chunks, return top-k.

Refactored from the old process_query.py: the `input()` and prompt.txt file-dump are
GONE. This is now a clean, importable, testable function — which is exactly what the
eval harness (roadmap Phase 3) and the API (Phase 5) will call.

ROADMAP Phase 2: this is the naive baseline ("dense" mode). You'll add sparse.py
(BM25), fusion.py (RRF), and rerank.py (cross-encoder) alongside it.
"""
import numpy as np
import pandas as pd
import requests
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from courselens.config import EMBEDDINGS_PATH, EMBEDDING_MODEL, OLLAMA_EMBED_URL, TOP_K

# Loaded once and cached so repeated queries don't re-read the file.
_df_cache: pd.DataFrame | None = None


def _load_index() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = joblib.load(EMBEDDINGS_PATH)
    return _df_cache


def embed_query(query: str) -> np.ndarray:
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBEDDING_MODEL, "input": [query]})
    r.raise_for_status()
    return np.array(r.json()["embeddings"][0]).reshape(1, -1)


def retrieve(query: str, k: int = TOP_K) -> pd.DataFrame:
    """Return the top-k most similar chunks as a DataFrame (_id, Video_id, text, start, end)."""
    df = _load_index()
    q = embed_query(query)
    sims = cosine_similarity(np.vstack(df["embedding"]), q).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return df.iloc[top_idx][["_id", "Video_id", "text", "start", "end"]]


if __name__ == "__main__":
    question = input("Ask a question: ")
    print(retrieve(question).to_json(orient="records", indent=2))
