# MENTOR ROADMAP — Turning this RAG project into a 12–15 LPA resume weapon

> **How to use this file:** Give this file to any AI assistant and say
> *"Act as my senior mentor. Follow MENTOR_ROADMAP.md. We are executing the 12-hour plan.
> Guide me phase by phase, verify each milestone before moving on."*
> The AI-mentor rules are at the bottom of this file. Everything the mentor needs —
> current-state audit, target architecture, hour-by-hour plan, acceptance criteria,
> resume bullets, and interview prep — is in here.

---

## PART 1 — Brutally honest assessment of the project as it stands today

### What the project currently is

A CLI pipeline that answers questions about a Hindi-language React video course:

```
videos/*.mp4
  → ffmpeg → audio/*.mp3                         (process_video.py)
  → Whisper large-v2, task=translate (hi→en)     (stt.py)      → jsons/*.json (segment chunks)
  → merge to ~400-word chunks, 50-word overlap   (merge_chunks.py) → merged_jsons/*.json
  → bge-m3 embeddings via local Ollama           (read_chunks.py)  → embeddings.joblib (pandas DF)
  → cosine similarity, top-5                     (process_query.py)
  → Gemini 2.0 Flash answer w/ video+timestamp citations (query_output.py) → Output.txt
```

### Verdict: **Level 1.5 out of 5.** Here is the brutal truth.

**The hard truth first:** this is a well-known YouTube-tutorial project. The exact
architecture — Hindi course videos → Whisper translate → bge-m3 on Ollama → joblib →
cosine → Gemini with timestamps — is a popular Hindi-creator tutorial that thousands of
students have on their resumes right now. An interviewer at ZS, Barclays, Michelin, or
ProcDNA who screens fresher/junior resumes has likely seen this exact project multiple
times. **As it stands, it does not differentiate you — it actively pattern-matches you
into the "followed a tutorial" bucket.**

**What is genuinely good (keep and amplify):**
- Real multimodal data engineering: video → audio → ASR → **cross-lingual** (Hindi speech
  → English answers). Most student RAGs are "PDF chatbots" — this data pipeline is more
  interesting than 90% of them.
- Timestamp-grounded citations (Video 03 | 25:12–27:18) — that's a legitimate
  grounding/anti-hallucination feature.
- Local embeddings (bge-m3) — shows you can work outside paid-API land.
- Sensible chunking instinct: merging Whisper's tiny segments into ~400-word overlapping
  chunks is the right idea.

**What is weak / missing (why it's Level 1.5, not 3):**

| # | Gap | Why an interviewer cares |
|---|-----|--------------------------|
| 1 | **Zero evaluation.** No metrics, no golden set, no way to say retrieval works. | This is THE gap. "How do you know your RAG is good?" is the first interview question, and today the honest answer is "I eyeballed it." Companies like ZS/Barclays care about measurable rigor above all. |
| 2 | **Naive retrieval.** Single dense embedding + cosine top-5. No keyword/BM25 signal, no reranking, no query handling. | Retrieval is the "R" in RAG; this is the 2023-tutorial baseline. Hybrid search + reranking is what industry actually ships. |
| 3 | **joblib DataFrame as the "vector store".** Full matrix loaded per query, brute-force cosine. | Signals no awareness of vector databases (Qdrant/Chroma/pgvector/FAISS), indexing, or metadata filtering. |
| 4 | **It's a script, not a system.** `input()` inside a function, prompt dumped to `prompt.txt`, answer to `Output.txt`, hardcoded relative paths, no error handling. | Nobody can run or demo it. No API = not a deployable artifact = "notebook-grade" work. |
| 5 | **No repo hygiene.** Not a git repo, `whisper-env/` virtualenv sitting inside the project, no README, no requirements.txt, no .gitignore, no config (API key handling ad hoc). | First 30 seconds on GitHub decides whether anyone reads further. A committed venv is an instant red flag. |
| 6 | **No story of decisions.** Why 400-word chunks? Why bge-m3? Why top-5? Currently: "the tutorial said so." | Interviews at this level are decision-defense sessions. Every number needs an ablation or a reason. |

### The one-line diagnosis

> You have a decent **data pipeline** wearing the costume of an AI project. What's missing
> is everything that makes it **engineering**: measurement, retrieval sophistication, a
> serving layer, and evidence.

---

## PART 2 — The chosen plan (one plan, committed)

Many upgrades are possible (agents, multimodal frame-search, fine-tuning, GraphRAG…).
**We are deliberately NOT doing those.** In 12 hours, breadth kills. The single highest-ROI
plan for 12–15 LPA data-science/ML roles is:

## ⭐ "CourseLens — an Evaluation-Driven, Hybrid-Retrieval RAG Service for Multilingual Video Courses"

**The pitch (memorize this):**
> "I built a cross-lingual RAG system over Hindi video lectures that answers in English
> with timestamp-grounded citations. Then I built an evaluation harness with a golden QA
> set and used it to systematically improve retrieval — going from naive dense search to
> hybrid BM25+dense with cross-encoder reranking — improving Recall@5 from X% to Y%.
> It's served as a FastAPI service backed by a vector database, fully Dockerized."

Why this plan beats the alternatives:

1. **Evaluation is the differentiator, not another feature.** 95% of student RAG projects
   have zero metrics. The moment you show a metrics table with an ablation
   (baseline → hybrid → +reranker), you jump out of the tutorial bucket. ZS and Barclays
   are analytics/quant-culture companies — *numbers on a resume bullet get interviews.*
2. **Hybrid retrieval + reranking is current industry standard practice**, not exotic.
   Interviewers recognize it and can probe it — and you'll have answers.
3. **A FastAPI + vector DB + Docker package** converts "script" into "system." That's the
   engineering maturity signal for 12–15 LPA.
4. It **respects the 12 hours** because it reuses everything you built (transcripts,
   chunks, embeddings pipeline) — we upgrade around it rather than rebuilding.

**Explicitly out of scope (do not let the mentor-AI or yourself scope-creep):**
frontend UI beyond a minimal demo (you already have a frontend project), agents,
fine-tuning, GraphRAG, auth, deployment to cloud, Kubernetes.

**Conditionally in scope:** multimodal (visual) retrieval — see the **Optional Stretch
Track: Multimodal Retrieval** in Part 4. It is NOT part of the core 12-hour commitment.
Only start it if Phases 0–7 are done with real time left over, or if a follow-up session
is used to extend the project. The core plan (text-only, evaluation-driven, hybrid
retrieval) must be fully working and measured first — multimodal is additive polish on
top of a working, proven system, never a replacement for one.

---

## PART 3 — Target architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                INGESTION (offline)           │
 videos/ ── ffmpeg ──▶ audio/ ── Whisper large-v2 (hi→en) ──▶ segment JSONs
                        │        └─ merge: ~400-word chunks, 50 overlap│
                        │        └─ bge-m3 embeddings (Ollama)         │
                        │        └─ upsert into ChromaDB (persistent)  │
                        │           + BM25 index (rank_bm25) over text │
                        └──────────────────────────────────────────────┘
                        ┌──────────────────────────────────────────────┐
                        │                QUERY (online, FastAPI)       │
  POST /ask ──▶ embed query ─▶ dense top-20 (Chroma)  ─┐
                └──────────▶ BM25 top-20               ─┤─ RRF fusion ─▶ top-10
                                                        │
                cross-encoder rerank (bge-reranker-base)│──▶ top-5 chunks
                                                        ▼
                Gemini 2.0 Flash + grounding prompt ──▶ answer + citations JSON
                        └──────────────────────────────────────────────┘
                        ┌──────────────────────────────────────────────┐
                        │                EVALUATION (offline)          │
  eval/golden_set.jsonl (≈40 QA pairs w/ source chunk ids)             │
    ─▶ retrieval metrics: Recall@k, Hit@1, MRR   (per retriever config)│
    ─▶ generation metrics: faithfulness & relevance via LLM-as-judge   │
    ─▶ results table written to eval/results.md  → pasted into README  │
                        └──────────────────────────────────────────────┘
       [ OPTIONAL — only if core plan is done early, see Part 4 stretch track ]
                        ┌──────────────────────────────────────────────┐
                        │        MULTIMODAL INGESTION (offline)        │
  videos/ ── ffmpeg (1 frame / N sec) ──▶ frames/*.jpg                 │
    ─▶ Tesseract OCR on frames (code/UI text on screen) ─▶ ocr text    │
    ─▶ embed OCR text (bge-m3) ─▶ upsert into Chroma, collection=      │
       "frames", metadata {video_id, timestamp, source="ocr"}         │
                        └──────────────────────────────────────────────┘
  At query time: dense search runs over BOTH collections ("transcript" and
  "frames"); results merged before fusion; citation shows source + timestamp
  so an answer can point to "code visible on screen" not just spoken words.
                        └──────────────────────────────────────────────┘
```

**Target repo layout** (restructure early — mentor should enforce this):

```
courselens/
├── README.md                  ← the real deliverable; metrics table lives here
├── requirements.txt
├── Dockerfile
├── .gitignore                 ← MUST exclude whisper-env/, videos/, audio/, *.joblib
├── .env.example               ← GOOGLE_API_KEY=...
├── src/courselens/
│   ├── config.py              ← paths, model names, k values — no magic numbers in code
│   ├── ingest/
│   │   ├── extract_audio.py   ← (from process_video.py)
│   │   ├── transcribe.py      ← (from stt.py)
│   │   ├── chunk.py           ← (from merge_chunks.py)
│   │   ├── index.py           ← embeds + upserts to Chroma, builds BM25 (from read_chunks.py)
│   │   └── extract_frames.py  ← [OPTIONAL, stretch] frame sampling + OCR → "frames" collection
│   ├── retrieval/
│   │   ├── dense.py           ← Chroma query
│   │   ├── sparse.py          ← BM25
│   │   ├── fusion.py          ← Reciprocal Rank Fusion
│   │   └── rerank.py          ← cross-encoder
│   ├── generation/
│   │   ├── prompts.py
│   │   └── answer.py          ← Gemini call, returns structured {answer, citations[]}
│   └── api/
│       └── main.py            ← FastAPI: POST /ask, GET /health
├── eval/
│   ├── golden_set.jsonl
│   ├── run_retrieval_eval.py
│   ├── run_generation_eval.py
│   └── results.md
└── data/                      ← gitignored; jsons/, merged_jsons/, chroma/
```

---

## PART 4 — The 12-hour execution plan (hour by hour)

Rules: timebox strictly. If a phase overruns by >30 min, cut its "nice-to-have" and move
on. Phases 3–4 (evaluation) are the ones you must never cut — they ARE the project.

### Phase 0 — Repo surgery & hygiene (Hour 0 → 1)
- `git init`; write `.gitignore` (whisper-env/, videos/, audio/, data/, *.joblib, .env,
  __pycache__, .DS_Store). **Delete/move `whisper-env` out of the repo** — use a fresh
  venv outside or `.venv` (gitignored).
- Restructure files into the layout above (move, don't rewrite yet). Add
  `requirements.txt` (pin: openai-whisper, chromadb, rank-bm25, sentence-transformers,
  fastapi, uvicorn, google-genai, numpy, pandas). If Part 4B (multimodal) is planned for
  this session, also add `pytesseract`, `imagehash`, `Pillow` and note that the Tesseract
  binary itself must be installed via the OS package manager (e.g. `brew install
  tesseract` on macOS) — `pip install pytesseract` alone only installs the Python wrapper.
- Create `config.py` with every constant (chunk size, overlap, k values, model names).
- First commit. GitHub repo created (private until done).
- ✅ **Acceptance:** `git log` shows clean initial commit; repo tree matches layout; no venv/media tracked.

### Phase 1 — Vector store + ingestion rewrite (Hour 1 → 2.5)
- Replace joblib-DataFrame with **ChromaDB (persistent local)**. `index.py`: read
  `merged_jsons/`, embed with bge-m3 (keep Ollama — it works), upsert with metadata
  `{video_id, start, end}`. Ids stable: `f"{video_id}_{chunk_id}"`.
- Build and pickle a **BM25 index** (`rank_bm25.BM25Okapi`) over tokenized chunk texts in
  the same script.
- Keep old JSONs as-is — do NOT re-run Whisper (hours of GPU time you don't have).
- ✅ **Acceptance:** ingestion runs end-to-end in one command; Chroma dir persists; a test
  query returns sensible chunks.

### Phase 2 — Hybrid retrieval + reranking (Hour 2.5 → 4.5)
- `dense.py`: Chroma top-20. `sparse.py`: BM25 top-20.
- `fusion.py`: **Reciprocal Rank Fusion**: `score(d) = Σ 1/(60 + rank_i(d))`. Take top-10.
- `rerank.py`: cross-encoder `BAAI/bge-reranker-base` via sentence-transformers
  (CPU is fine for 10 pairs) → final top-5.
- Expose one function `retrieve(query, mode="dense"|"hybrid"|"hybrid+rerank")` — the
  eval harness will ablate over `mode`.
- ✅ **Acceptance:** all three modes return ranked chunk ids for a query; you can articulate
  why BM25 catches what dense misses (exact terms: "useEffect", "Appwrite", error names).

### Phase 3 — Golden set + retrieval evaluation ⭐ CORE (Hour 4.5 → 7)
- Build `eval/golden_set.jsonl`: **~40 questions**, each `{question,
  expected_chunk_ids: [...], answerable: true|false}`.
  - Fast method: for ~25 chunks spread across videos, have an LLM draft a question whose
    answer is in that chunk; **you hand-verify/edit every one** (this is what makes it
    "golden"). Add ~8 questions answerable from multiple chunks, and ~7 unanswerable/
    off-topic questions (for refusal testing).
- `run_retrieval_eval.py`: for each mode compute **Recall@5, Hit@1 (precision of top
  result), MRR@10**. Output a markdown table to `eval/results.md`.
- Run the ablation: `dense` vs `hybrid` vs `hybrid+rerank`. **Record the numbers.** If
  hybrid doesn't beat dense, investigate tokenization/k values — that debugging session
  itself becomes interview material.
- ✅ **Acceptance:** a real table like:

  | Retriever | Recall@5 | Hit@1 | MRR@10 |
  |---|---|---|---|
  | Dense (bge-m3) | 0.xx | 0.xx | 0.xx |
  | Hybrid (RRF) | 0.xx | 0.xx | 0.xx |
  | Hybrid + reranker | 0.xx | 0.xx | 0.xx |

### Phase 4 — Generation quality + faithfulness eval (Hour 7 → 8.5)
- Rewrite the prompt in `prompts.py`: ask Gemini to return **structured JSON**
  `{"answer": str, "citations": [{"video_id", "start", "end"}], "grounded": bool}` —
  parse and format timestamps in code, not in the prompt (more reliable).
- Refusal path: unanswerable questions must yield a clean "not in the material" response.
- `run_generation_eval.py`: LLM-as-judge (Gemini judging Gemini is acceptable; note the
  caveat) scoring **faithfulness** (is every claim supported by retrieved chunks?) and
  **answer relevance** on ~15 golden questions, plus refusal accuracy on the 7
  unanswerable ones. Log scores to `eval/results.md`.
- ✅ **Acceptance:** faithfulness/relevance numbers exist; all unanswerable questions refuse.

### Phase 5 — FastAPI service + Docker (Hour 8.5 → 10)
- `api/main.py`: `POST /ask {"question": ..., "mode": ...}` → `{answer, citations,
  retrieved_chunks, latency_ms}`; `GET /health`. Load indexes once at startup. Basic
  error handling (Ollama down, empty retrieval).
- `Dockerfile` (python-slim; document that Ollama runs on host). `.env.example`.
- ✅ **Acceptance:** `uvicorn` up; a `curl` to `/ask` returns a cited answer; latency logged.

### Phase 6 — README + evidence (Hour 10 → 11.5)
The README is what gets read in the 45 seconds a recruiter/interviewer gives you. Must contain:
1. One-paragraph pitch (cross-lingual, timestamp-grounded, evaluation-driven).
2. Architecture diagram (ASCII from Part 3 is fine, or a simple image).
3. **The metrics tables** (retrieval ablation + generation eval) — front and center.
4. "Design decisions" section: chunk size rationale, why bge-m3 (multilingual — matches
   Hindi-origin content), why RRF over weighted-sum, why cross-encoder reranks better
   than bi-encoder retrieval, LLM-judge caveats.
5. Quickstart: 4 commands (ingest → eval → serve → curl example w/ real output).
6. A short demo: sample Q/A with citations (screenshot or code block); GIF optional.
- ✅ **Acceptance:** a stranger can understand and run the project from README alone.

### Phase 7 — Ship (Hour 11.5 → 12)
- Final commit history sane (5–10 meaningful commits, not one blob). Push public.
- Write the resume bullets (Part 5) with your REAL numbers filled in.

### Stretch goals — ONLY if genuinely ahead of schedule (in priority order)
1. Query rewriting (LLM reformulates the query before retrieval; add as 4th eval mode).
2. Semantic caching of repeated queries (dict + embedding similarity ≥0.95).
3. Streamlit one-pager demo (30 min max).
4. GitHub Actions CI running the retrieval eval on push.
5. **Multimodal retrieval — see dedicated track below.** This is the biggest and most
   "exciting" stretch item; do it only after 1–4 above are either done or deliberately
   skipped, and only if the CORE plan (Phases 0–7) is fully working and measured.

---

## PART 4B — Optional Stretch Track: Multimodal (Visual) Retrieval

**Why this exists:** the core plan (Parts 2–4 above) upgrades *how well* you search text
— it doesn't change *what* you can search. Right now the system only "hears" the video;
it never "sees" it. Code written on screen, UI being demoed, diagrams, error messages
shown visually — all of that is invisible to a transcript-only RAG. This track adds a
second retrieval source (OCR'd video frames) so the system can answer questions like
*"which video shows the useEffect cleanup function on screen?"* — something no
transcript-only system can do. This is the one addition in this whole roadmap that's a
genuinely new **capability**, not just better engineering of the existing one.

**Time budget: 2–2.5 hours.** Treat this as a strict add-on block, done only after the
core system (text RAG + eval + API) already works end-to-end. Never let this block eat
into Phase 3/4 (evaluation) time — evaluation is still what gets you the interview;
multimodal is what makes the interview memorable.

**Do NOT attempt this if:**
- Phases 0–5 aren't done and demoed working, OR
- You're already past hour 10 in a single 12-hour session (do it in a follow-up session
  instead — it's fully separable and doesn't touch anything already shipped).

### Step-by-step build

**Step M1 — Frame extraction (~20 min)**
- For each video, sample 1 frame every ~10–15 seconds with ffmpeg (don't do every second —
  8 videos × dense sampling is thousands of near-duplicate images for no benefit):
  `ffmpeg -i videos/01.mp4 -vf fps=1/12 frames/01/frame_%04d.jpg`
- Store frames as `frames/{video_id}/frame_NNNN.jpg`; compute each frame's timestamp from
  its index × interval (no need to parse ffmpeg logs).
- Skip near-duplicate frames cheaply: hash consecutive frames (e.g. average-hash via
  `imagehash` library) and drop a frame if it's near-identical to the previous kept
  frame — this cuts OCR work a lot on static "talking head + code editor" footage.

**Step M2 — OCR each kept frame (~30–40 min)**
- Use **Tesseract** (`pytesseract`) — free, local, no API cost, good enough for on-screen
  code/text. Run OCR per frame; keep only frames where OCR text has, say, ≥15
  alphanumeric characters (filters blank/noise frames — talking-head-only frames with no
  on-screen text should be dropped, not indexed).
- Output one JSON per video: `frames/{video_id}_ocr.json` = list of
  `{video_id, timestamp, frame_path, ocr_text}`.
- **Be honest about OCR quality in the README** — Tesseract on video-captured code (small
  font, syntax-highlighted, possibly blurry) will be noisy. This is fine and expected;
  note it as a known limitation, don't hide it. (If time allows and quality is bad,
  mention EasyOCR or a vision-LLM caption as a documented alternative you considered.)

**Step M3 — Index OCR text as a second collection (~20 min)**
- Reuse the exact same embedding function (bge-m3) — embed each `ocr_text` chunk (you may
  want to merge consecutive frames' OCR text within a shot before embedding, similar
  logic to `merge_chunks.py`, to avoid one embedding per near-duplicate frame).
- Upsert into a **separate Chroma collection** named `"frames"` (keep it separate from
  `"transcript"` — don't blend embeddings of two very different text distributions into
  one collection; separate collections let you weight/filter by source later).
- Metadata per entry: `{video_id, timestamp, source: "ocr"}`.

**Step M4 — Merge into retrieval (~30–40 min)**
- At query time, run dense search against **both** collections (`"transcript"` and
  `"frames"`), each returning top-k with their own scores.
- Combine before/alongside the existing RRF fusion step — simplest approach: treat the
  frame-collection results as a third ranked list into the same RRF formula used for
  BM25+dense (Part 4, Phase 2). This reuses code you already wrote instead of inventing a
  new fusion mechanism.
- Update citation formatting: an answer sourced from a frame should say
  `(Video 03 | 18:42 — on-screen code)` vs the existing `(Video 03 | 25:12–27:18 —
  spoken)`, so it's visibly clear which modality grounded which claim.
- Update the prompt in `generation/prompts.py` to tell the model it may receive both
  spoken-transcript context and on-screen-text context, and to cite which kind it used.

**Step M5 — Extend the eval set, don't build a separate one (~15–20 min)**
- Add 5–8 questions to the SAME `eval/golden_set.jsonl` from Phase 3 that are answerable
  only from on-screen text (e.g. an exact variable name or import statement visible only
  in a code editor shot, not spoken aloud). Mark them
  `{"expected_source": "ocr", ...}`.
- Re-run `run_retrieval_eval.py` with the frame collection included; add a row to
  `eval/results.md`: **"Hybrid + rerank + multimodal"** — show whether Recall@5 on the
  *OCR-only-answerable* subset improved versus the text-only system (which should score
  near 0 on those, by construction — a clean, honest before/after story).

### Acceptance criteria for this track
- At least one real question in your demo is answerable ONLY because of the frame/OCR
  index (i.e., not present anywhere in the spoken transcript) — this is your proof it's
  not decorative.
- `eval/results.md` shows the OCR-subset Recall@5 comparison (text-only vs +multimodal).
- README has a short "Multimodal Retrieval" section explaining the OCR limitation
  honestly and the design decision to use separate collections + RRF reuse.

### What this adds to your pitch
> "...and additionally indexed on-screen visual content — sampling frames and OCR'ing
> code/UI text — as a second retrieval collection fused via the same RRF mechanism,
> letting the system answer questions the audio transcript alone cannot, such as
> retrieving the exact video timestamp where specific code appears on screen."

### Interview questions this opens up (add to Part 6 prep)
- Why a separate Chroma collection instead of one mixed index? → Transcript prose and
  OCR'd code/UI text have very different lexical/semantic distributions; mixing them in
  one embedding space risks one modality drowning out the other in similarity search;
  separate collections keep retrieval quality diagnosable per source and let you weight
  or filter by modality later.
- Why sample frames every ~10–15s instead of every second? → Talking-head/course footage
  changes on-screen content slowly; dense sampling multiplies OCR cost for near-zero
  information gain; near-duplicate hashing further prunes redundant frames.
- Why Tesseract and not a vision-LLM caption? → Cost/speed/local-first tradeoff — Tesseract
  is free and fast for text-heavy frames (code, UI labels); a vision-LLM would likely be
  more accurate but adds per-frame API cost and latency, worth naming as a future upgrade.
- How would you scale frame indexing to thousands of videos? → Sample+hash-dedupe at
  ingestion, batch OCR on GPU/parallel workers, store frame images in object storage
  (S3-like) with only OCR text + pointer in the vector DB — never store raw images in the
  vector store itself.

---

## PART 5 — Resume bullets (fill in real numbers after Phase 3/4)

> **CourseLens — Evaluation-Driven Cross-Lingual RAG for Video Courses** *(Python, Whisper, ChromaDB, FastAPI, Gemini)*
> - Built an end-to-end RAG system over N hours of Hindi video lectures: Whisper-based
>   speech-to-text translation, overlap chunking, and multilingual (bge-m3) embeddings in
>   ChromaDB, answering English queries with timestamp-grounded video citations.
> - Engineered hybrid retrieval (BM25 + dense with Reciprocal Rank Fusion) and
>   cross-encoder reranking, improving Recall@5 from __% to __% and MRR from __ to __ on a
>   hand-verified 40-question golden set.
> - Designed an automated evaluation harness (retrieval metrics + LLM-as-judge
>   faithfulness scoring) enabling measured iteration; achieved __% refusal accuracy on
>   out-of-scope queries to control hallucination.
> - Productionized as a Dockerized FastAPI service with structured JSON citations and
>   p50 latency of __ ms.

*Optional 5th bullet, only if Part 4B (multimodal) is actually built and measured:*
> - Extended retrieval beyond the audio transcript by sampling video frames and OCR'ing
>   on-screen code/UI text into a second indexed collection, fused into the same ranking
>   pipeline — enabling retrieval of visually-grounded content invisible to transcript-only
>   RAG, validated on a dedicated eval subset.

Never write a bullet you can't defend for 5 minutes of follow-up questions.

---

## PART 6 — Interview defense prep (study during breaks / after the build)

You WILL be asked these. The project only pays off if you can answer them.

**Retrieval / IR**
- Why hybrid? → Dense embeddings miss exact lexical matches (API names like `useEffect`,
  `Appwrite`, error strings); BM25 misses paraphrase/semantic matches; RRF combines rank
  information without needing score calibration between the two systems.
- Why RRF over weighted score sum? → BM25 and cosine scores live on different,
  query-dependent scales; rank-based fusion is scale-free and robust, no tuning needed.
- Bi-encoder vs cross-encoder? → Bi-encoder embeds query & doc independently (fast,
  indexable, approximate). Cross-encoder attends over the concatenated pair (accurate,
  expensive) — hence retrieve-many-cheap, rerank-few-expensive.
- Why cosine similarity? Why 400-word chunks with overlap? → Have your ablation/reasoning:
  too-small chunks lose context for generation; too-large dilute the embedding and blow
  the context budget; overlap prevents answers straddling a boundary from being lost.

**Evaluation**
- How do you know it works? → Golden set + Recall@5 / MRR for retrieval; LLM-judge
  faithfulness for generation; refusal tests for hallucination. Know your numbers cold.
- Weaknesses of LLM-as-judge? → Self-preference bias, prompt sensitivity, no ground truth;
  mitigate with human spot-checks (you hand-verified the golden set) and by judging
  against retrieved context, not world knowledge.
- What's Recall@5 vs MRR? → Recall@5: fraction of questions where a gold chunk appears in
  top-5. MRR: average of 1/rank of the first gold chunk — rewards ranking it higher.

**System / scaling**
- 10,000 videos instead of 8? → Move Chroma → Qdrant/pgvector with HNSW; metadata
  filtering by course; batch/async ingestion; GPU inference server or API for embeddings;
  cache hot queries.
- Why did you translate to English at ASR time instead of doing Hindi retrieval? → Single
  language simplifies the index and the answer language; bge-m3 is multilingual so
  cross-lingual queries still work; tradeoff: translation errors propagate — mention it.
- Hallucination control? → Grounded prompt + citations required per claim + explicit
  refusal path + faithfulness eval to measure it.

**Honesty questions**
- "Did you follow a tutorial?" → "The initial transcription pipeline started from a
  tutorial pattern; everything that makes it a system — hybrid retrieval, reranking, the
  evaluation harness, the API, the ablations — I designed and built myself, and the
  metrics in the README are from my own golden set." (This is honest AND strong.)

---

## PART 7 — Rules for the AI mentor executing this plan

If you are an AI assistant reading this file, you are this developer's **senior mentor**,
not a code generator. Follow these rules:

1. **Follow the phase order in Part 4.** Announce each phase, its time budget, and its
   acceptance criteria before starting. Verify acceptance criteria (actually run the
   code/eval) before moving to the next phase.
2. **Enforce the timebox.** If a phase overruns, cut nice-to-haves and say so explicitly.
   Never add out-of-scope items (UI polish, agents, cloud deploys) even if asked casually
   — remind the developer of Part 2's scope contract.
3. **Explain every decision's tradeoff in 2–3 sentences** as you build (this doubles as
   interview prep). After each phase, quiz the developer with 2 questions from Part 6.
4. **Evaluation is sacred.** Phases 3–4 must produce real numbers in `eval/results.md`.
   If the numbers are bad, debug retrieval — do not fudge, do not skip. A negative
   ablation result honestly reported is still interview gold.
5. **The golden set must be human-verified.** Draft questions with an LLM if needed, but
   require the developer to read and approve every one.
6. **Keep commits meaningful** (one per milestone, imperative messages). README metrics
   must come from actually-run evals, never invented.
7. **Don't re-run Whisper transcription** — reuse `jsons/` / `merged_jsons/`. That compute
   is already spent.
8. Target quality bar: a hiring manager at ZS/Barclays skims the GitHub repo for 60
   seconds — README metrics table, architecture diagram, clean src/ layout must land in
   that window.

---

## Level map (where you are, where each phase takes you)

- **L1 — tutorial script** *(you are here)*: works on your machine, no metrics, no repo.
- **L2 — clean project**: git, structure, config, README, vector DB. *(Phases 0–1)*
- **L3 — real retrieval engineering**: hybrid + reranking with justified choices. *(Phase 2)*
- **L4 — measured system**: golden set, ablation table, faithfulness eval — top ~5% of
  fresher RAG projects. *(Phases 3–4)*
- **L5 — deployable service**: FastAPI + Docker + evidence-rich README — the 12–15 LPA
  signal. *(Phases 5–7)*
- **L6 — multimodal, measured** *(optional, Part 4B)*: frame + OCR retrieval fused into
  the same pipeline, with its own eval subset proving it adds real capability, not
  decoration. This is the one item on this list that's a genuinely new capability rather
  than better engineering of the same one — do it only after L5 is solid.
- **L7+ (beyond 12h, optional later)**: query rewriting eval'd, CI-run evals, semantic
  caching, Qdrant + filtering, monitoring dashboards, multi-course tenancy, vision-LLM
  captions in place of Tesseract OCR.
