# CourseLens

Cross-lingual RAG over Hindi-language React course videos: transcribes + translates
speech to English (Whisper), retrieves relevant transcript chunks, and answers questions
in English with **timestamp-grounded video citations** (e.g. `Video 03 | 25:12 – 27:18`).

> This repo is the **clean starting point**. The working baseline (ingest → dense
> retrieval → Gemini answer) runs today. The engineering that makes it a *system* —
> hybrid retrieval, reranking, an evaluation harness, and a FastAPI service — is what you
> build next, phase by phase. The full plan lives in **`MENTOR_ROADMAP.md`**; the honest
> starting-state audit is in **`CURRENT_STATE.md`**.

## Project structure

```
src/courselens/
├── config.py            # all paths, model names, k-values — no magic numbers in code
├── ingest/
│   ├── extract_audio.py # ffmpeg: videos -> audio            (WORKING)
│   ├── transcribe.py    # Whisper hi->en -> data/jsons        (WORKING)
│   ├── chunk.py         # merge into ~400-word chunks         (WORKING)
│   └── index.py         # bge-m3 embeddings -> joblib          (WORKING)
├── retrieval/
│   ├── dense.py         # cosine top-k baseline               (WORKING)
│   ├── sparse.py        # BM25                                (TODO — Phase 2)
│   ├── fusion.py        # Reciprocal Rank Fusion              (TODO — Phase 2)
│   └── rerank.py        # cross-encoder reranker              (TODO — Phase 2)
├── generation/
│   ├── prompts.py       # grounded prompt builder             (WORKING)
│   └── answer.py        # retrieve -> prompt -> Gemini         (WORKING)
└── api/
    └── main.py          # FastAPI /ask, /health               (TODO — Phase 5)
eval/                    # golden set + metrics harness         (TODO — Phase 3/4 ⭐)
data/                    # videos, audio, jsons, merged_jsons, index (gitignored where large)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # makes `import courselens` work everywhere
pip install -r requirements.txt
cp .env.example .env        # then add your GOOGLE_API_KEY
```

External dependencies (not pip-installable):
- **ffmpeg** on PATH — audio extraction
- **Ollama** running locally with `bge-m3` pulled: `ollama pull bge-m3` — embeddings
- **Google Gemini API key** in `.env` as `GOOGLE_API_KEY` — answer generation

## Run the working baseline

```bash
# Ingest (transcripts for the processed videos already exist — these skip finished work)
python -m courselens.ingest.extract_audio     # videos -> audio
python -m courselens.ingest.transcribe        # audio  -> data/jsons        (slow; skips done)
python -m courselens.ingest.chunk             # jsons  -> data/merged_jsons
python -m courselens.ingest.index             # chunks -> data/index/embeddings.joblib

# Ask a question (needs Ollama + GOOGLE_API_KEY)
python -m courselens.generation.answer
```

## What to build next

Follow `MENTOR_ROADMAP.md` in order. The stub files already mark every build target with
its phase and target signature:
1. **Phase 1** — swap joblib for ChromaDB; build the BM25 index in `ingest/index.py`.
2. **Phase 2** — `retrieval/{sparse,fusion,rerank}.py`.
3. **Phase 3/4 ⭐** — the evaluation harness in `eval/` (this is the differentiator).
4. **Phase 5** — the FastAPI service in `api/main.py`, then Docker.
5. **Phase 6** — paste the real `eval/results.md` metrics tables into this README.
