"""Configuration for the PDF RAG pipeline (FAISS + Cohere).

All tunables live here. Secrets are read from the environment (loaded from a
local `.env` file when present) and are never hard-coded.

Author:
    Prakhar Johri — https://www.linkedin.com/in/johriprakhar/
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from <project>/.env if it exists.
load_dotenv(BASE_DIR / ".env")


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env_str(key, str(default)))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
# Required at runtime. Set it in .env as COHERE_API_KEY=...
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# --------------------------------------------------------------------------- #
# Source document
# --------------------------------------------------------------------------- #
PDF_PATH = BASE_DIR / _env_str("PDF_PATH", "Rhea_resume.pdf")

# Value stored in the `source` column for every chunk (used for citations).
DOCUMENT_SOURCE = _env_str("DOCUMENT_SOURCE", "sample-pdf")

# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
VECTOR_STORE_DIR = BASE_DIR / _env_str("VECTOR_STORE_DIR", "Vector_Store")
INDEX_PATH = VECTOR_STORE_DIR / _env_str("INDEX_FILENAME", "vector_db.index")
DOCS_CSV_PATH = VECTOR_STORE_DIR / _env_str("DOCS_FILENAME", "docs.csv")

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
CHUNK_SIZE = _env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)

# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME = _env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
TOP_K = _env_int("TOP_K", 5)

# Max acceptable L2 distance for the nearest chunk. Anything above this is
# treated as "not covered by the document".
DISTANCE_THRESHOLD = _env_float("DISTANCE_THRESHOLD", 1.7)

# --------------------------------------------------------------------------- #
# Cohere chat
# --------------------------------------------------------------------------- #
COHERE_CHAT_MODEL = _env_str("COHERE_CHAT_MODEL", "command-a-03-2025")
COHERE_TEMPERATURE = _env_float("COHERE_TEMPERATURE", 0.3)

PROMPT_TEMPLATE = _env_str(
    "PROMPT_TEMPLATE",
    "Based on the document content: {context}, answer the question: '{question}'",
)

# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
IRRELEVANT_QUERY_MESSAGE = _env_str(
    "IRRELEVANT_QUERY_MESSAGE", "Please ask a relevant question."
)


def validate() -> None:
    """Fail fast with a clear message if required settings are missing."""
    if not COHERE_API_KEY:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Add it to your .env file "
            "(see env.example) or export it in your shell."
        )
