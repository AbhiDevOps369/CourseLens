"""Central configuration — every path, model name, and magic number lives here.

Roadmap rule (Phase 0): "no magic numbers in code". When you need to tune chunk
size, k values, or swap a model, change it HERE, not scattered across scripts.
"""
from pathlib import Path

# Project root = three levels up from this file (src/courselens/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]

# --- Data locations -------------------------------------------------------
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"            # raw .mp4 course videos (gitignored)
AUDIO_DIR = DATA_DIR / "audio"             # extracted .mp3 (gitignored)
SEGMENTS_DIR = DATA_DIR / "jsons"          # per-video Whisper segment transcripts
CHUNKS_DIR = DATA_DIR / "merged_jsons"     # merged ~400-word overlapping chunks
INDEX_DIR = DATA_DIR / "index"             # embeddings / vector store (gitignored)
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.joblib"

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
GEMINI_MODEL = "gemini-2.0-flash"

# --- Retrieval ------------------------------------------------------------
TOP_K = 5
