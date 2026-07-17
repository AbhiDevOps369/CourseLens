# SceneScribe

*(package name in code: `courselens` — SceneScribe is the project's public name)*

An evaluation-driven, hybrid-retrieval RAG service over Hindi-language video courses.
Speech is transcribed and translated to English (Whisper), indexed with both dense
(multilingual embeddings) and sparse (BM25) retrieval, fused with Reciprocal Rank Fusion,
and answered by Gemini with **timestamp-grounded video citations**
(e.g. `Video 03 | 25:22 – 27:36`) — plus a refusal path for out-of-scope questions and an
LLM-judge harness that measures faithfulness and relevance instead of assuming them.

Every number in this README comes from an actually-run evaluation script in `eval/` —
none of it is estimated or invented. Where something didn't work as expected (a
reranker regression, a failed refusal-threshold idea, an overly strict judge), that's
reported honestly below instead of hidden, because the investigation itself is part of
the engineering.

## Architecture

```
                    INGESTION (offline)
videos/*.mp4 -> ffmpeg -> audio/*.mp3 -> Whisper large-v2 (hi->en, translate)
    -> merge into ~400-word overlapping chunks (50-word overlap)
    -> bge-m3 embeddings (local Ollama) -> ChromaDB (persistent)
    -> BM25 index (rank_bm25) over the same chunk text

                    QUERY (online, FastAPI)
POST /ask {question, mode} -> embed query
    -> dense top-20 (Chroma)  ---\
    -> BM25 top-20             ---+--> Reciprocal Rank Fusion -> top-k
    -> [optional] cross-encoder rerank (evaluated, NOT shipped — see below)
    -> Gemini + grounded prompt -> {answer, citations, retrieved_chunks,
                                     latency_ms, grounded}

                    EVALUATION (offline, eval/)
golden_set.jsonl (42 hand-verified Q&A pairs: 35 answerable, 7 unanswerable)
    -> run_retrieval_eval.py   -> Recall@5 / Hit@1 / MRR@10 per retriever mode
    -> run_refusal_eval.py     -> refusal recall / false-refusal rate
    -> run_generation_eval.py  -> LLM-judge faithfulness / relevance
    -> inspect_faithfulness.py -> manual spot-check tool for judge verdicts
```

## Results

### Retrieval ablation

| Retriever          | Recall@5 | Hit@1 | MRR@10 |
|---------------------|----------|-------|--------|
| Dense (bge-m3)       | 0.771    | 0.400 | 0.536  |
| **Hybrid (RRF)** ⭐  | 0.771    | **0.571** | **0.652** |
| Hybrid + reranker    | 0.771    | 0.486 | 0.614  |

**Hybrid (BM25 + dense, fused via RRF) is what ships.** The cross-encoder reranker
(`BAAI/bge-reranker-base`) was built and evaluated, not assumed — it *regressed* Hit@1
(0.571 → 0.486) and added ~5x latency per request (hybrid: ~2.0s observed vs.
hybrid+rerank: ~10.4s observed, single-request samples). Root cause, found via a
dedicated diagnostic script rather than guessed: the reranker is trained on clean,
well-formed text and consistently confused topically-adjacent chunks in this corpus's
noisy, conversational ASR-translated transcript — a real domain mismatch, not a config
error. Shipping hybrid-without-rerank was the evidence-based call.

### Generation quality (sampled, reproducible seed)

| Metric                     | Score |
|-----------------------------|-------|
| Refusal recall (caught off-topic questions) | **7/7 = 1.000** |
| False-refusal rate (wrongly blocked real ones) | 1/35 = 0.029 |
| Faithfulness (LLM-judge, raw)   | 9/14 = 0.643 |
| Answer relevance (LLM-judge)    | 14/14 = 1.000 |

**Refusal** is prompt-based, not a raw-score threshold — a score-threshold gate on
RRF-fused scores was investigated and rejected: RRF scores are rank-relative (there's
always a "rank 1" result even when every candidate is irrelevant), so an off-topic
question like *"who won the FIFA World Cup"* scored **identically** to genuinely relevant
questions (0.0328, the mathematical maximum for a chunk ranked 1st by both dense and
sparse search). That's not a usable confidence signal. The system instead relies on a
single, consistent refusal phrase built into the generation prompt, validated against
the golden set: 100% recall on the 7 unanswerable questions, 1 false refusal out of 35
answerable ones (which traces back to that question's gold chunk missing the top-5 in
retrieval — Recall@5 is 0.771, so this is consistent with, not independent of, the
retrieval numbers above).

**Faithfulness** uses Gemini to judge Gemini's own answers — a known bias
(self-preference: a model tends to grade its own output generously). The raw score is
0.643, but manually reading the flagged "unfaithful" cases against their source context
(`eval/inspect_faithfulness.py`) showed the judge penalizing valid multi-chunk synthesis
and paraphrasing as if it were invented information, not catching actual hallucination —
one case correctly synthesized five different source chunks into a coherent answer with
four out of five citations verified byte-accurate, and was still marked unfaithful. The
true faithfulness rate is very likely higher than 0.643; this is reported as-is rather
than adjusted, with the caveat stated plainly instead of hidden.

## Design decisions

- **~400-word chunks, 50-word overlap.** Small enough to stay topically coherent,
  large enough to preserve context for generation; overlap prevents an answer that
  straddles a chunk boundary from being lost entirely.
- **bge-m3 embeddings.** Multilingual by design — matches Hindi-origin spoken content
  even though everything is translated to English at ASR time.
- **RRF over a weighted score sum.** BM25 and cosine similarity live on different,
  query-dependent scales with no natural calibration between them; RRF combines *rank*
  information instead of raw scores, so it needs no tuning and is robust to that
  scale mismatch. (The same rank-only property is exactly why it fails as a refusal
  signal — see above.)
- **Translate-at-ASR-time, not retrieve-in-Hindi.** Simplifies the index and the answer
  language to one; bge-m3 being multilingual means cross-lingual queries still work.
  Tradeoff, stated honestly: translation errors from Whisper propagate downstream and
  are not corrected later in the pipeline.
- **Cross-encoder reranker: built, evaluated, not shipped.** See the retrieval ablation
  above — kept as an evaluated, documented option, not deleted, since the finding (why
  it regressed) is itself the interesting part.

## Honesty note

The initial transcription pipeline followed a common tutorial pattern for this kind of
project. Everything that makes it an evaluated *system* — hybrid retrieval with a
measured ablation, the reranker investigation, the refusal-threshold investigation, the
golden set, the refusal path, the FastAPI service, and the LLM-judge harness with its
own documented failure mode — was designed, built, debugged, and evaluated from here.

## What's not built yet

Explicitly out of scope for this phase: agents, fine-tuning, GraphRAG, auth, cloud
deployment, Kubernetes.

**Multimodal (OCR) retrieval** is a planned, separately-scheduled extension, not part of
this build — indexing on-screen visual text (code, UI) as a second retrieval source.
A small pilot script (`pilot_ocr.py`) exists to test whether on-screen text on this
specific video content (live coding, heavily narrated) adds information beyond what's
already in the spoken transcript, before committing to the full pipeline.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env        # add your GOOGLE_API_KEY
```

External dependencies (not pip-installable):
- **ffmpeg** on PATH — audio extraction
- **Ollama** running locally with `bge-m3` pulled — embeddings
- **Tesseract** (`brew install tesseract`) — only needed for the OCR pilot

## Run it

```bash
# Ingest (skips already-processed videos)
python -m courselens.ingest.extract_audio
python -m courselens.ingest.transcribe
python -m courselens.ingest.chunk
python -m courselens.ingest.index          # builds both Chroma + BM25 indexes

# Evaluate (writes real numbers to eval/results.md)
python -m eval.run_retrieval_eval
python -m eval.run_refusal_eval
python -m eval.run_generation_eval

# Serve
uvicorn courselens.api.main:app --reload
```

## Example

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What will I learn from this course", "mode": "hybrid"}'
```

```json
{
  "answer": "Based on the provided course material, the following projects are covered:\n\n* Todo Context and Local Storage Project: focuses on business logic and functionality using the Context API and local storage (Video 03 | 34:12 - 36:14).\n* Mega Project (Full-Fledged App): a complex, production-grade application covering authentication, CRUD, third-party API handling, and deployment (Video 03 | 25:22 - 27:36).",
  "citations": [
    {"video_id": "03", "start": 2052, "end": 2174},
    {"video_id": "01", "start": 0, "end": 118},
    {"video_id": "03", "start": 2732, "end": 2868}
  ],
  "grounded": true,
  "latency_ms": 1962.93
}
```
