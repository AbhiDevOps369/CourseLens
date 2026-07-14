"""Ingest step 4 — build BOTH retrieval indexes from data/merged_jsons/.

Replaces the old joblib-DataFrame store (roadmap Phase 1). One command now:
  1. embeds each merged chunk with bge-m3 (Ollama) and upserts it into a
     persistent ChromaDB collection with stable ids + metadata, and
  2. tokenizes the same chunk texts into a rank_bm25 Okapi index, pickled to disk.

Stable id = f"{Video_id}_{_id}"  (e.g. "01_3") — deliberately matches the id
format the eval golden set references, so Phase 3 can score retrieval directly.

Requires Ollama running locally with bge-m3 pulled (`ollama pull bge-m3`).
Run:  python -m courselens.ingest.index
"""
import json
import pickle
import re

import requests
import chromadb
from rank_bm25 import BM25Okapi

from courselens.config import (
    CHUNKS_DIR, INDEX_DIR, CHROMA_DIR, TRANSCRIPT_COLLECTION, BM25_PATH,
    EMBEDDING_MODEL, OLLAMA_EMBED_URL,
)


def create_embedding(text_list):
    """Batch-embed a list of strings via Ollama's bge-m3."""
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBEDDING_MODEL, "input": text_list})
    r.raise_for_status()
    return r.json()["embeddings"]


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — shared by ingest (build) and sparse.py (query)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks() -> list[dict]:
    """Flatten all merged_jsons into one list of records with stable ids."""
    records = []
    for file in sorted(CHUNKS_DIR.glob("*.json")):
        content = json.loads(file.read_text())
        for c in content["merged_chunks"]:
            records.append({
                "id": f'{c["Video_id"]}_{c["_id"]}',
                "video_id": c["Video_id"],
                "start": float(c["start"]),
                "end": float(c["end"]),
                "text": c["text"],
            })
    return records


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records = load_chunks()
    print(f"loaded {len(records)} chunks from {CHUNKS_DIR}")

    # --- 1. Dense: embed + upsert into a persistent Chroma collection --------
    print("embedding chunks with bge-m3 (Ollama) ...")
    embeddings = create_embedding([r["text"] for r in records])

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Recreate cleanly so re-running ingest is idempotent, not additive.
    try:
        client.delete_collection(TRANSCRIPT_COLLECTION)
    except Exception:
        pass
    col = client.create_collection(TRANSCRIPT_COLLECTION, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[r["id"] for r in records],
        embeddings=embeddings,
        documents=[r["text"] for r in records],
        metadatas=[{"video_id": r["video_id"], "start": r["start"], "end": r["end"]} for r in records],
    )
    print(f"  -> Chroma collection '{TRANSCRIPT_COLLECTION}' persisted to {CHROMA_DIR} ({col.count()} docs)")

    # --- 2. Sparse: build + pickle a BM25 index over the same texts ----------
    print("building BM25 index ...")
    tokenized = [tokenize(r["text"]) for r in records]
    bm25 = BM25Okapi(tokenized)
    # Persist the model AND the id/metadata order so sparse.py can map ranks -> chunks.
    meta = [{"id": r["id"], "video_id": r["video_id"], "start": r["start"],
             "end": r["end"], "text": r["text"]} for r in records]
    BM25_PATH.write_bytes(pickle.dumps({"bm25": bm25, "meta": meta}))
    print(f"  -> BM25 index pickled to {BM25_PATH} ({len(meta)} docs)")

    print("Ingestion complete. Both dense (Chroma) and sparse (BM25) indexes are ready.")


if __name__ == "__main__":
    main()