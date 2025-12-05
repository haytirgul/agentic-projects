"""Global settings and configuration for the repo analyst project.

This module loads environment variables and provides configuration constants
that can be imported throughout the project.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Import hardcoded constants
try:
    from .const import (
        REPO_EXCLUDE_PATTERNS,
        VECTOR_DB_COLLECTION_NAME,
        DEFAULT_MAX_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_MAX_CHUNK_TOKENS,
        TEXT_EXTENSIONS,
        CODE_EXTENSIONS,
        MAX_CHUNK_SIZE_MARKDOWN,
        CHUNK_OVERLAP_MARKDOWN,
    )
except ImportError:
    from const import (
        REPO_EXCLUDE_PATTERNS,
        VECTOR_DB_COLLECTION_NAME,
        DEFAULT_MAX_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        DEFAULT_MAX_CHUNK_TOKENS,
        TEXT_EXTENSIONS,
        CODE_EXTENSIONS,
        MAX_CHUNK_SIZE_MARKDOWN,
        CHUNK_OVERLAP_MARKDOWN,
    )

# Load environment variables from .env file
load_dotenv()

__all__ = [
    # Project paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "HTTPX_REPO_DIR",
    "INDEX_DIR",
    "VECTOR_DB_DIR",
    "RAG_COMPONENTS_DIR",

    # Repository settings
    "REPO_EXCLUDE_PATTERNS",

    # Chunking settings
    "CHUNK_STRATEGY",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DEFAULT_MAX_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MAX_CHUNK_TOKENS",
    "MAX_CHUNK_SIZE_MARKDOWN",
    "CHUNK_OVERLAP_MARKDOWN",

    # Embedding settings
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "SENTENCE_TRANSFORMERS_HOME",

    # Retrieval settings
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    "MAX_ITERATIONS",
    "VECTOR_WEIGHT",
    "SPARSE_WEIGHT",
    "RERANKING_ENABLED",

    # Vector store settings
    "VECTOR_DB_PERSIST_DIRECTORY",
    "VECTOR_DB_COLLECTION_NAME",

    # Conversation memory settings
    "MAX_HISTORY_TURNS",
    "ENABLE_CONVERSATION_MEMORY",

    # LLM settings
    "GOOGLE_API_KEY",
    "LLM_MODEL",
    "MODEL_FAST",
    "MODEL_INTERMEDIATE",
    "MODEL_SLOW",
    "LEVEL_TO_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_TOP_P",
    "LLM_TOP_K",

    # LangSmith observability
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",

    # Output settings
    "LOG_LEVEL",
    "SHOW_RETRIEVAL_LOG",
    "VERBOSE_OUTPUT",
    "ENABLE_STREAMING",
    "LAZY_INITIALIZATION",

    # File extensions
    "TEXT_EXTENSIONS",
    "CODE_EXTENSIONS",
]

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
HTTPX_REPO_DIR = DATA_DIR / os.getenv("HTTPX_REPO_DIR", "httpx")
INDEX_DIR = DATA_DIR / "index"
VECTOR_DB_DIR = DATA_DIR / "vector_db"
RAG_COMPONENTS_DIR = DATA_DIR / "rag_components"

# ============================================================================
# REPOSITORY SETTINGS
# ============================================================================

# Repository exclude patterns imported from const.py

# ============================================================================
# CHUNKING SETTINGS
# ============================================================================

# Chunking strategy: "ast" or "window"
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "ast")

# Maximum chunk size (lines for window, N/A for AST)
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "500"))

# Overlap between chunks (for window strategy)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding model name
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Sentence Transformers cache directory
SENTENCE_TRANSFORMERS_HOME = os.getenv("SENTENCE_TRANSFORMERS_HOME", str(DATA_DIR / "sentence_transformers_cache"))
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_HOME

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Number of top results to retrieve
TOP_K = int(os.getenv("TOP_K", "10"))

# Similarity threshold for filtering results
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Maximum number of retrieval iterations
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# Hybrid search weights
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))

# Enable reranking
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"

# ============================================================================
# VECTOR STORE SETTINGS
# ============================================================================

# Vector database settings (FAISS)
VECTOR_DB_PERSIST_DIRECTORY = str(VECTOR_DB_DIR)
# VECTOR_DB_COLLECTION_NAME imported from const.py

# ============================================================================
# CONVERSATION MEMORY SETTINGS
# ============================================================================

# Maximum conversation history turns to maintain
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# Enable conversation memory
ENABLE_CONVERSATION_MEMORY = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"

# ============================================================================
# LLM SETTINGS
# ============================================================================

# Google API Key (required)
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not configured. Please set it in your .env file."
    )

# Model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# Model Tiers: fast, intermediate, slow
MODEL_FAST = os.getenv("MODEL_FAST", "gemini-2.5-flash-lite")
MODEL_INTERMEDIATE = os.getenv("MODEL_INTERMEDIATE", "gemini-2.5-flash")
MODEL_SLOW = os.getenv("MODEL_SLOW", "gemini-2.5-pro")

LEVEL_TO_MODEL = {
    "fast": MODEL_FAST,
    "intermediate": MODEL_INTERMEDIATE,
    "slow": MODEL_SLOW,
}

# LLM parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000")) if os.getenv("LLM_MAX_TOKENS") else None
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))  # seconds
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))

# ============================================================================
# LANGSMITH OBSERVABILITY (OPTIONAL)
# ============================================================================

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "code-rag-agent")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Display options
SHOW_RETRIEVAL_LOG = os.getenv("SHOW_RETRIEVAL_LOG", "true").lower() == "true"
VERBOSE_OUTPUT = os.getenv("VERBOSE_OUTPUT", "false").lower() == "true"
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

# Optimization
LAZY_INITIALIZATION = os.getenv("LAZY_INITIALIZATION", "true").lower() == "true"

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

# Processing constants imported from const.py

# ============================================================================
# OUTPUT FILE PATHS
# ============================================================================

# Processed data output files
CODE_CHUNKS_FILE = DATA_DIR / "processed" / "code_chunks.json"
MARKDOWN_CHUNKS_FILE = DATA_DIR / "processed" / "markdown_chunks.json"
TEXT_CHUNKS_FILE = DATA_DIR / "processed" / "text_chunks.json"
ALL_CHUNKS_FILE = DATA_DIR / "processed" / "all_chunks.json"

# Processing configuration
MAX_CHUNK_SIZE_TEXT = DEFAULT_MAX_CHUNK_SIZE
# MAX_CHUNK_SIZE_MARKDOWN and CHUNK_OVERLAP_MARKDOWN imported from const.py

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure directories exist
HTTPX_REPO_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
RAG_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

# Validate chunk strategy
if CHUNK_STRATEGY not in ["ast", "window"]:
    raise ValueError(f"Invalid CHUNK_STRATEGY: {CHUNK_STRATEGY}. Must be 'ast' or 'window'")

# Validate weights sum to 1.0
if abs(VECTOR_WEIGHT + SPARSE_WEIGHT - 1.0) > 0.01:
    raise ValueError("VECTOR_WEIGHT + SPARSE_WEIGHT must equal 1.0")
