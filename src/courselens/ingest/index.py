"""Ingest step 4 — embed merged chunks with bge-m3 (Ollama) -> data/index/embeddings.joblib.

Fixes the old path bug (was written to CWD, read from Codes/): the artifact now
always lives at config.EMBEDDINGS_PATH so ingest and retrieval agree.

Requires Ollama running locally with bge-m3 pulled (`ollama pull bge-m3`).
Run:  python -m courselens.ingest.index

ROADMAP Phase 1: this joblib-DataFrame is the baseline "vector store". You will
replace it with a real vector DB (ChromaDB, persistent) + a BM25 index here.
"""
import json

import requests
import pandas as pd
import joblib

from courselens.config import CHUNKS_DIR, INDEX_DIR, EMBEDDINGS_PATH, EMBEDDING_MODEL, OLLAMA_EMBED_URL


def create_embedding(text_list):
    r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBEDDING_MODEL, "input": text_list})
    r.raise_for_status()
    return r.json()["embeddings"]


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for file in sorted(CHUNKS_DIR.glob("*.json")):
        print(f"embedding {file.name} ...")
        content = json.loads(file.read_text())
        merged_chunks = content["merged_chunks"]
        embeddings = create_embedding([c["text"] for c in merged_chunks])
        for chunk, emb in zip(merged_chunks, embeddings):
            chunk["embedding"] = emb
            all_chunks.append(chunk)

    df = pd.DataFrame.from_records(all_chunks)
    joblib.dump(df, EMBEDDINGS_PATH)
    print(f"Embedding complete -> {EMBEDDINGS_PATH} ({len(df)} chunks).")


if __name__ == "__main__":
    main()
