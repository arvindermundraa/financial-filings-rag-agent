"""Paths and settings. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(os.getenv("FINRAG_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.getenv("FINRAG_DATA_DIR", ROOT / "data"))
PDF_DIR = DATA_DIR / "pdfs"
QUESTIONS_PATH = DATA_DIR / "financebench_open_source.jsonl"
PAGES_PATH = DATA_DIR / "pages.jsonl"
INDEX_DIR = Path(os.getenv("FINRAG_INDEX_DIR", DATA_DIR / "index"))

FINANCEBENCH_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"

# Chunking: filing pages are long tables; ~250-word windows with overlap keep a table row
# and its column headers together more often than not.
CHUNK_WORDS = int(os.getenv("FINRAG_CHUNK_WORDS", 250))
CHUNK_OVERLAP = int(os.getenv("FINRAG_CHUNK_OVERLAP", 50))

# Dense (LSA) vector size.
LSA_DIM = int(os.getenv("FINRAG_LSA_DIM", 256))

# Reciprocal rank fusion constant (60 is the value from Cormack et al., 2009).
RRF_K = int(os.getenv("FINRAG_RRF_K", 60))

LLM_MODEL = os.getenv("FINRAG_MODEL", "claude-sonnet-5-5")
AGENT_MAX_STEPS = int(os.getenv("FINRAG_MAX_STEPS", 6))
