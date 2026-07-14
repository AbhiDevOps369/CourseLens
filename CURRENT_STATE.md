# CURRENT_STATE — What exists in this project right now (as of 2026-07-07)

> **Purpose of this file:** This is the "where I am now" snapshot. Feed it to an AI
> assistant *together with* `MENTOR_ROADMAP.md` (the "where I'm going" plan) and say:
> *"Here is the current state of my project (CURRENT_STATE.md) and the plan
> (MENTOR_ROADMAP.md). Act as my senior mentor and take me from the current state to the
> target, phase by phase."*
>
> This file describes reality — the actual code, data, and gaps that exist today. It does
> NOT describe the target design (that's the roadmap). Keep this file honest and factual.
> When something changes, update this file so the AI always knows the true starting point.

---

## 1. One-line summary of what this project does today

A local command-line pipeline that takes Hindi-language React course videos, transcribes
+ translates them to English with Whisper, embeds the transcript, and answers typed
questions using dense (cosine) retrieval + Google Gemini, returning answers with video +
timestamp citations. It runs as a sequence of standalone scripts, not as an application.

**Source course:** "Chai Aur React" (Chai Aur Code / Hitesh Choudhary) — Hindi-language
React tutorial videos.

---

## 2. The actual pipeline, step by step (what really runs today)

```
videos/*.mp4  (18 raw video files present)
     │
     │  process_video.py  →  ffmpeg extracts audio
     ▼
audio/*.mp3   (8 files currently extracted — NOT all 18)
     │
     │  stt.py  →  Whisper "large-v2", language="hi", task="translate"  (Hindi → English)
     ▼
jsons/*.json  (8 files)  — per-video segment chunks: {_id, Video_id, start, end, text} + full_text
     │
     │  merge_chunks.py  →  merges tiny Whisper segments into ~400-word chunks, 50-word overlap
     ▼
merged_jsons/*.json  (8 files)  — {_id, Video_id, start, end, text} merged chunks + full_text
     │
     │  read_chunks.py  →  bge-m3 embeddings via local Ollama (http://localhost:11434)
     ▼
embeddings.joblib  — a pandas DataFrame with an "embedding" column, saved via joblib
     │
     │  process_query.py  →  input() a question, embed it, cosine similarity vs all rows, take top-5
     │                       builds a big prompt string, writes it to prompt.txt
     ▼
     │  query_output.py  →  sends prompt to Gemini 2.0 Flash (google-genai), writes answer
     ▼
Output.txt  — the final answer with (Video NN | MM:SS – MM:SS) citations
```

**How you actually run it today:** manually, one script at a time, from the project root,
in this order: `process_video.py` → `stt.py` → `merge_chunks.py` → `read_chunks.py` →
`query_output.py` (which internally calls `process_query.py` and prompts you to type a
question via `input()`).

---

## 3. File-by-file inventory (what each file is)

### Code (`Codes/`)
| File | What it does | Notes / issues |
|------|--------------|----------------|
| `process_video.py` | ffmpeg: every file in `videos/` → `audio/<name>.mp3` | Names come out as `01.mp4.mp3`; no error handling; processes whatever is in `videos/`. |
| `stt.py` | Whisper large-v2, Hindi→English translate, per-segment chunks → `jsons/*.json` | The real ASR step. Slow (large-v2). Already run for 8 videos. |
| `merge_chunks.py` | Merges segments into ~400-word overlapping chunks → `merged_jsons/*.json` | Good chunking logic. `MAX_WORDS=400`, `OVERLAP_WORDS=50`. Minor bug: `buffer_start` is reset to the *current* chunk's start after a flush rather than tracking the overlap region's start — worth reviewing. |
| `read_chunks.py` | Embeds merged chunks with bge-m3 (Ollama) → `embeddings.joblib` | Writes `embeddings.joblib` to CWD, but `process_query.py` loads it from `Codes/embeddings.joblib` — path inconsistency to watch. |
| `process_query.py` | `input()` a question, embed, cosine top-5, build prompt, write `prompt.txt` | Retrieval lives here. This is the "R" that the roadmap upgrades most. `input()` inside a function makes it un-importable/un-testable cleanly. |
| `query_output.py` | Calls `create_prompt()`, sends to Gemini 2.0 Flash → `Output.txt` | Uses `GOOGLE_API_KEY` from env. Prompt is duplicated/wrapped redundantly. |
| `rename.py` | One-off: fixes `.mp3` filenames in `audio/` | Utility scratch script. |
| `test.py` | One-off Whisper test on a single `audio/sample.mp3` | Scratch/experiment file; uses old `Video_number` key. Not part of the pipeline. |
| `embeddings.joblib` | Saved DataFrame of embedded chunks | Binary artifact currently committed alongside code. |
| `prompt.txt` (in Codes/) | A saved prompt copy | Stray artifact. |

### Data
| Folder / file | Contents | Count |
|---------------|----------|-------|
| `videos/` | Raw `.mp4` course videos | **18 files** |
| `audio/` | Extracted `.mp3` | **8 files** (only 8 of 18 done) |
| `jsons/` | Per-video Whisper transcripts (segments) | 8 |
| `merged_jsons/` | Merged ~400-word chunks | 8 |
| `prompt.txt` (root) | Last generated prompt | 1 |
| `Output.txt` (root) | Last generated answer | 1 |

### Environment
| Item | State |
|------|-------|
| `whisper-env/` | A Python 3.14 virtualenv committed **inside** the project (should be removed/gitignored). |
| Ollama | Expected running locally on `:11434` serving `bge-m3` (external dependency, not in repo). |
| Gemini | `google-genai` SDK, model `gemini-2.0-flash`, needs `GOOGLE_API_KEY` env var. |
| Git | **Not a git repository.** No `.gitignore`, no README, no `requirements.txt`. |

---

## 4. What genuinely works right now (verified capabilities)

- End-to-end: you can ask a React question and get a coherent English answer with
  video/timestamp citations (see `Output.txt` for a real sample output).
- Cross-lingual transcription (Hindi speech → English text) via Whisper large-v2.
- Sensible chunk merging with overlap.
- Local embeddings through Ollama/bge-m3 (no paid embedding API).
- Timestamp citation formatting in the answer.

---

## 5. What does NOT exist yet (the honest gap list)

These are the gaps the roadmap is designed to close. Listed here so the AI mentor knows
exactly what's missing at the starting line:

1. **No evaluation of any kind** — no golden question set, no Recall/MRR/faithfulness
   metrics. Quality is currently judged by eyeballing `Output.txt`.
2. **No hybrid retrieval / reranking** — retrieval is a single dense cosine top-5. No
   BM25/keyword signal, no cross-encoder reranker, no query preprocessing.
3. **No real vector store** — embeddings live in a `joblib` pandas DataFrame; every query
   brute-force cosines the whole matrix. No Chroma/FAISS/Qdrant/pgvector.
4. **No serving layer** — it's scripts with `input()` and file dumps (`prompt.txt`,
   `Output.txt`). No FastAPI, no `/ask` endpoint, no way for anyone else to run it.
5. **No repo hygiene** — not a git repo; `whisper-env/` committed; no README, no
   `requirements.txt`, no `.gitignore`, no config module (magic numbers/paths inline).
6. **No multimodal / visual retrieval** — only the audio transcript is searchable; nothing
   reads on-screen code/UI from the video frames. (Optional stretch in the roadmap.)
7. **Data incomplete** — only 8 of 18 videos are processed through the pipeline.
8. **Structural issues** — `input()` inside a function, inconsistent artifact paths
   (`embeddings.joblib` written to CWD but read from `Codes/`), scratch files
   (`test.py`, `rename.py`) mixed with pipeline code, no error handling anywhere.

---

## 6. Known bugs / rough edges to be aware of

- **Audio filenames**: `process_video.py` produces `01.mp4.mp3` (double extension);
  `rename.py` was a manual fix. The pipeline elsewhere assumes clean `01.mp3` / `01.json`.
- **`embeddings.joblib` path**: written to current directory by `read_chunks.py` but
  loaded from `Codes/embeddings.joblib` by `process_query.py`. Works only if you run from
  a specific directory.
- **`merge_chunks.py` overlap start**: after flushing a chunk, `buffer_start` is set to the
  current chunk's `start`, which doesn't match the retained overlap words' true start time
  — timestamps on post-first chunks may be slightly off. Review during Phase 1.
- **`test.py`** uses an older schema (`Video_number` instead of `Video_id`) — it's dead
  code, safe to delete.
- **Prompt duplication**: `query_output.py` wraps the already-complete prompt from
  `process_query.py` in another instruction block — redundant, clean up in Phase 4.

---

## 7. External dependencies the environment assumes

- **ffmpeg** installed on the system (for audio extraction / future frame extraction).
- **Ollama** running locally with the `bge-m3` model pulled (`ollama pull bge-m3`).
- **Whisper** (`openai-whisper`) with the `large-v2` model weights (large download).
- **Google Gemini API key** in env as `GOOGLE_API_KEY`.
- Python packages currently used ad hoc (no requirements file yet): `openai-whisper`,
  `google-genai`, `requests`, `pandas`, `numpy`, `scikit-learn`, `joblib`.

---

## 8. Where this sits on the roadmap's level map

Per `MENTOR_ROADMAP.md`, this project is at **Level 1.5 / 5** — "tutorial script": it
works on the developer's machine, but has no metrics, no real retrieval engineering, no
vector DB, no serving layer, and no repo hygiene. The roadmap's Phase 0 is the immediate
next action (git init, restructure, kill the committed venv).

---

## 9. Suggested first message to the AI mentor

> "I'm at the state described in CURRENT_STATE.md. Follow MENTOR_ROADMAP.md. Start with
> Phase 0 (repo surgery & hygiene). Before writing code, confirm my environment
> (ffmpeg, Ollama+bge-m3, Whisper, GOOGLE_API_KEY) and tell me the exact acceptance
> criteria for Phase 0. Do NOT re-run Whisper — reuse the existing jsons/ and
> merged_jsons/. Decide with me whether to process the remaining 10 videos now or ship
> the eval-driven system on the 8 already done first."
