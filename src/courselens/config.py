"""Central configuration — every path, model name, and magic number lives here.

Roadmap rule (Phase 0): "no magic numbers in code". When you need to tune chunk
size, k values, or swap a model, change it HERE, not scattered across scripts.
"""
from pathlib import Path

from dotenv import load_dotenv

# Load .env into this process's environment (os.environ) as soon as config.py
# is imported — which happens before almost anything else, since every module
# depends on config. This replaces manually `export`-ing GOOGLE_API_KEY in
# whatever terminal happens to be open; now it loads the same way every time,
# from every entry point (CLI scripts, uvicorn, eval scripts).
load_dotenv()

# Project root = three levels up from this file (src/courselens/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]

# --- Data locations -------------------------------------------------------
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"            # raw .mp4 course videos (gitignored)
AUDIO_DIR = DATA_DIR / "audio"             # extracted .mp3 (gitignored)
SEGMENTS_DIR = DATA_DIR / "jsons"          # per-video Whisper segment transcripts
CHUNKS_DIR = DATA_DIR / "merged_jsons"     # merged ~400-word overlapping chunks
INDEX_DIR = DATA_DIR / "index"             # embeddings / vector store (gitignored)

# --- Vector store (Phase 1) ----------------------------------------------
# ChromaDB replaces the old joblib DataFrame. Persistent = survives restarts,
# loaded once, HNSW-indexed instead of brute-force cosine over the whole matrix.
CHROMA_DIR = INDEX_DIR / "chroma"          # persistent Chroma store (gitignored)
TRANSCRIPT_COLLECTION = "transcript"       # spoken-transcript chunks

# --- Sparse index (Phase 1 build, Phase 2 query) -------------------------
# rank_bm25 Okapi index over tokenized chunk texts, pickled alongside Chroma.
BM25_PATH = INDEX_DIR / "bm25.pkl"

# --- Chunking (ingest/chunk.py) ------------------------------------------
MAX_WORDS = 400
OVERLAP_WORDS = 50

# --- Transcription (ingest/transcribe.py) --------------------------------
WHISPER_MODEL = "large-v2"
WHISPER_LANGUAGE = "hi"       # source audio language
WHISPER_TASK = "translate"    # translate -> English

# --- Embeddings (ingest/index.py, retrieval/dense.py) --------------------
EMBEDDING_MODEL = "bge-m3"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"

# --- Generation (generation/answer.py) -----------------------------------
GEMINI_MODEL = "gemini-3.1-flash-lite"

# --- Retrieval ------------------------------------------------------------
TOP_K = 5