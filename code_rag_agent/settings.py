"""Configuration settings for the Code RAG Agent.

This module provides a unified configuration system with two categories:

1. SYSTEM CONSTANTS: Fixed values that define system behavior (not user-configurable)
   - Schema definitions, regex patterns, file extensions
   - Algorithm constants (RRF_K, embedding dimensions)

2. USER SETTINGS: Configurable via environment variables (.env file)
   - API keys, model names, feature flags
   - Tunable parameters (temperatures, chunk sizes, top-k)

Author: Hay Hoffman
Version: 2.0
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# SYSTEM CONSTANTS - Not user-configurable
# ============================================================================

# -----------------------------------------------------------------------------
# Project Paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
RAG_COMPONENTS_DIR = DATA_DIR / "rag_components"

# Output file paths
CODE_CHUNKS_FILE = DATA_DIR / "processed" / "code_chunks.json"
MARKDOWN_CHUNKS_FILE = DATA_DIR / "processed" / "markdown_chunks.json"
TEXT_CHUNKS_FILE = DATA_DIR / "processed" / "text_chunks.json"
ALL_CHUNKS_FILE = DATA_DIR / "processed" / "all_chunks.json"

# Retrieval paths (for hybrid retriever)
CHUNKS_DIR = DATA_DIR / "processed"  # Directory containing chunk JSON files

# -----------------------------------------------------------------------------
# Schema Definitions
# -----------------------------------------------------------------------------

# Source types for unified chunk schema
SOURCE_TYPES: list[str] = ["code", "markdown", "text"]

# Valid chunk types for each source type
CHUNK_TYPE_MAP: dict[str, list[str]] = {
    "code": ["function", "class"],
    "markdown": ["markdown_section", "markdown_section_chunk"],
    "text": ["text_file", "text_file_chunk"],
}

# File extensions by category
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".toml", ".yml", ".yaml", ".txt", ".ini", ".cfg", ".conf", ".json", ".pyi"
})
CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".cs", ".php", ".rb", ".go", ".rs"
})

# -----------------------------------------------------------------------------
# Algorithm Constants
# -----------------------------------------------------------------------------

# RRF (Reciprocal Rank Fusion) - Algorithm constant, not tunable
RRF_K: int = 60

# FAISS embedding dimension (determined by embedding model)
FAISS_EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2

# BM25 stopwords (linguistic constant)
BM25_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been"
})

# -----------------------------------------------------------------------------
# Fast Path Router Patterns
# -----------------------------------------------------------------------------

FAST_PATH_PATTERNS: dict[str, str] = {
    "hex_code": r"^0x[0-9A-Fa-f]+$",
    "function_name": r"^[a-z_][a-z0-9_]*$",
    "camel_case_class": r"^[A-Z][a-zA-Z0-9]*$",
    "file_name": r"^[\w\-]+\.[a-z]{2,4}$",
}

# -----------------------------------------------------------------------------
# Repository Exclude Patterns
# -----------------------------------------------------------------------------

REPO_EXCLUDE_PATTERNS: list[str] = [
    "__pycache__/*",
    ".git/*",
    "docs/img/*",
    "docs/css/*",
    "docs/overrides/*",
    "docs/CNAME",
    "scripts/build",
    "scripts/clean",
    "scripts/publish",
    "scripts/sync-version",
    "*.pyc",
    "*.pyo",
    "*.egg-info/*",
    ".pytest_cache/*",
    ".coverage",
    "htmlcov/*",
]


# ============================================================================
# USER SETTINGS - Configurable via environment variables
# ============================================================================

# -----------------------------------------------------------------------------
# API Keys (Required)
# -----------------------------------------------------------------------------

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not configured. Please set it in your .env file."
    )

# -----------------------------------------------------------------------------
# LangSmith Observability (Optional)
# -----------------------------------------------------------------------------

LANGCHAIN_TRACING_V2: bool = (
    os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
)
LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "code-rag-agent")
LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

# Model tiers (can override via env)
# Note: gemini-2.0-flash has higher free tier limits (1500 RPD vs 20 RPD for flash-lite)
MODEL_FAST: str = os.getenv("MODEL_FAST", "gemini-2.0-flash-lite")
MODEL_INTERMEDIATE: str = os.getenv("MODEL_INTERMEDIATE", "gemini-2.0-flash")
MODEL_SLOW: str = os.getenv("MODEL_SLOW", "gemini-2.5-pro")

LEVEL_TO_MODEL: dict[str, str] = {
    "fast": MODEL_FAST,
    "intermediate": MODEL_INTERMEDIATE,
    "slow": MODEL_SLOW,
}

# Embedding model
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# Sentence Transformers cache
SENTENCE_TRANSFORMERS_HOME: str = os.getenv(
    "SENTENCE_TRANSFORMERS_HOME", str(DATA_DIR / "sentence_transformers_cache")
)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_HOME

# -----------------------------------------------------------------------------
# LLM Parameters
# -----------------------------------------------------------------------------

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
_llm_max_tokens_env = os.getenv("LLM_MAX_TOKENS")
LLM_MAX_TOKENS: int | None = int(_llm_max_tokens_env) if _llm_max_tokens_env else None
LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_TOP_K: int = int(os.getenv("LLM_TOP_K", "40"))

# -----------------------------------------------------------------------------
# Intent Classification Configuration
# -----------------------------------------------------------------------------

INTENT_MODEL: str = MODEL_FAST  # Use fast model for intent classification
INTENT_TEMPERATURE: float = float(os.getenv("INTENT_TEMPERATURE", "0.0"))
INTENT_MAX_TOKENS: int = int(os.getenv("INTENT_MAX_TOKENS", "1024"))

# -----------------------------------------------------------------------------
# Router Configuration
# -----------------------------------------------------------------------------

ROUTER_MODEL: str = MODEL_FAST  # Always use fast model for routing
ROUTER_TEMPERATURE: float = float(os.getenv("ROUTER_TEMPERATURE", "0.1"))
ROUTER_MAX_TOKENS: int = int(os.getenv("ROUTER_MAX_TOKENS", "1024"))
ROUTER_MAX_TREE_DEPTH: int = int(os.getenv("ROUTER_MAX_TREE_DEPTH", "4"))
FAST_PATH_ROUTER_ENABLED: bool = (
    os.getenv("FAST_PATH_ROUTER_ENABLED", "true").lower() == "true"
)

# -----------------------------------------------------------------------------
# Synthesis Configuration
# -----------------------------------------------------------------------------

SYNTHESIS_MODEL: str = MODEL_INTERMEDIATE  # Use intermediate for quality
SYNTHESIS_TEMPERATURE: float = float(os.getenv("SYNTHESIS_TEMPERATURE", "0.3"))
SYNTHESIS_MAX_TOKENS: int = int(os.getenv("SYNTHESIS_MAX_TOKENS", "4096"))

# -----------------------------------------------------------------------------
# Retrieval Configuration
# -----------------------------------------------------------------------------

# Top-K settings
TOP_K_PER_REQUEST: int = int(os.getenv("TOP_K_PER_REQUEST", "5"))
RRF_TOP_N: int = int(os.getenv("RRF_TOP_N", "50"))
MAX_RETRIEVAL_REQUESTS: int = int(os.getenv("MAX_RETRIEVAL_REQUESTS", "5"))

# RRF weights (semantic vs keyword)
RRF_BM25_WEIGHT: float = float(os.getenv("RRF_BM25_WEIGHT", "0.4"))
RRF_VECTOR_WEIGHT: float = float(os.getenv("RRF_VECTOR_WEIGHT", "1.0"))


# -----------------------------------------------------------------------------
# Context Expansion Limits
# -----------------------------------------------------------------------------

MAX_RELATED_METHODS: int = int(os.getenv("MAX_RELATED_METHODS", "3"))
MAX_IMPORTS: int = int(os.getenv("MAX_IMPORTS", "5"))
MAX_CHILDREN_SECTIONS: int = int(os.getenv("MAX_CHILDREN_SECTIONS", "10"))

# -----------------------------------------------------------------------------
# BM25 Tokenization
# -----------------------------------------------------------------------------

BM25_MIN_TOKEN_LENGTH: int = int(os.getenv("BM25_MIN_TOKEN_LENGTH", "3"))
BM25_SPLIT_CAMELCASE: bool = os.getenv("BM25_SPLIT_CAMELCASE", "true").lower() == "true"
BM25_SPLIT_SNAKE_CASE: bool = (
    os.getenv("BM25_SPLIT_SNAKE_CASE", "true").lower() == "true"
)
BM25_KEEP_HEX_CODES: bool = os.getenv("BM25_KEEP_HEX_CODES", "true").lower() == "true"

# -----------------------------------------------------------------------------
# Chunking Configuration
# -----------------------------------------------------------------------------

CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "ast")
MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))  # characters
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters
MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "1000"))  # tokens

# Derived settings (use MAX_CHUNK_TOKENS as single source of truth)
MAX_CHUNK_SIZE_MARKDOWN: int = MAX_CHUNK_TOKENS  # For markdown processing
MAX_CHUNK_SIZE_TEXT: int = MAX_CHUNK_SIZE  # For text file processing
CHUNK_OVERLAP_MARKDOWN: int = CHUNK_OVERLAP  # For markdown recursive splits


# -----------------------------------------------------------------------------
# Repository Configuration
# -----------------------------------------------------------------------------

HTTPX_REPO_DIR: Path = DATA_DIR / os.getenv("HTTPX_REPO_DIR", "httpx")
# -----------------------------------------------------------------------------
# Conversation Memory
# -----------------------------------------------------------------------------

MAX_HISTORY_TURNS: int = int(os.getenv("MAX_HISTORY_TURNS", "5"))
ENABLE_CONVERSATION_MEMORY: bool = os.getenv(
    "ENABLE_CONVERSATION_MEMORY", "true"
).lower() == "true"

# -----------------------------------------------------------------------------
# Feature Flags
# -----------------------------------------------------------------------------

SECURITY_ENABLED: bool = os.getenv("SECURITY_ENABLED", "true").lower() == "true"
ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
LAZY_INITIALIZATION: bool = os.getenv("LAZY_INITIALIZATION", "true").lower() == "true"

# -----------------------------------------------------------------------------
# Security Settings
# -----------------------------------------------------------------------------

SECURITY_FAIL_CLOSED: bool = os.getenv("SECURITY_FAIL_CLOSED", "true").lower() == "true"
SECURITY_MAX_BATCH_SIZE: int = int(os.getenv("SECURITY_MAX_BATCH_SIZE", "32"))
SECURITY_MODEL_CACHE_DIR: str = os.getenv(
    "SECURITY_MODEL_CACHE_DIR", str(DATA_DIR / "security_models")
)
SECURITY_USE_FP16: bool = os.getenv("SECURITY_USE_FP16", "false").lower() == "true"

# -----------------------------------------------------------------------------
# Output Settings
# -----------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
SHOW_RETRIEVAL_LOG: bool = os.getenv("SHOW_RETRIEVAL_LOG", "true").lower() == "true"
VERBOSE_OUTPUT: bool = os.getenv("VERBOSE_OUTPUT", "false").lower() == "true"


# ============================================================================
# VALIDATION & INITIALIZATION
# ============================================================================

# Ensure directories exist
HTTPX_REPO_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
RAG_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

# Validate chunk strategy
if CHUNK_STRATEGY not in ("ast", "window"):
    raise ValueError(f"Invalid CHUNK_STRATEGY: {CHUNK_STRATEGY}. Must be 'ast' or 'window'")



# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # === SYSTEM CONSTANTS ===
    # Paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "INDEX_DIR",
    "VECTOR_DB_DIR",
    "RAG_COMPONENTS_DIR",
    "CODE_CHUNKS_FILE",
    "MARKDOWN_CHUNKS_FILE",
    "TEXT_CHUNKS_FILE",
    "ALL_CHUNKS_FILE",
    "CHUNKS_DIR",
    # Schema
    "SOURCE_TYPES",
    "CHUNK_TYPE_MAP",
    "TEXT_EXTENSIONS",
    "CODE_EXTENSIONS",
    # Algorithm constants
    "RRF_K",
    "FAISS_EMBEDDING_DIM",
    "BM25_STOPWORDS",
    "FAST_PATH_PATTERNS",
    "REPO_EXCLUDE_PATTERNS",
    # === USER SETTINGS ===
    # API Keys
    "GOOGLE_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
    # Models
    "MODEL_FAST",
    "MODEL_INTERMEDIATE",
    "MODEL_SLOW",
    "LEVEL_TO_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "SENTENCE_TRANSFORMERS_HOME",
    # LLM Parameters
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_TOP_P",
    "LLM_TOP_K",
    # Intent Classification
    "INTENT_MODEL",
    "INTENT_TEMPERATURE",
    "INTENT_MAX_TOKENS",
    # Router
    "ROUTER_MODEL",
    "ROUTER_TEMPERATURE",
    "ROUTER_MAX_TOKENS",
    "ROUTER_MAX_TREE_DEPTH",
    "FAST_PATH_ROUTER_ENABLED",
    # Synthesis
    "SYNTHESIS_MODEL",
    "SYNTHESIS_TEMPERATURE",
    "SYNTHESIS_MAX_TOKENS",
    # Retrieval
    "TOP_K_PER_REQUEST",
    "RRF_TOP_N",
    "MAX_RETRIEVAL_REQUESTS",
    "RRF_BM25_WEIGHT",
    "RRF_VECTOR_WEIGHT",
    # Context Expansion
    "MAX_RELATED_METHODS",
    "MAX_IMPORTS",
    "MAX_CHILDREN_SECTIONS",
    # BM25
    "BM25_MIN_TOKEN_LENGTH",
    "BM25_SPLIT_CAMELCASE",
    "BM25_SPLIT_SNAKE_CASE",
    "BM25_KEEP_HEX_CODES",
    # Chunking
    "CHUNK_STRATEGY",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "MAX_CHUNK_TOKENS",
    "MAX_CHUNK_SIZE_MARKDOWN",
    "MAX_CHUNK_SIZE_TEXT",
    "CHUNK_OVERLAP_MARKDOWN",
    # Repository
    "HTTPX_REPO_DIR",
    # Conversation
    "MAX_HISTORY_TURNS",
    "ENABLE_CONVERSATION_MEMORY",
    # Feature Flags
    "SECURITY_ENABLED",
    "ENABLE_STREAMING",
    "LAZY_INITIALIZATION",
    # Security Settings
    "SECURITY_FAIL_CLOSED",
    "SECURITY_MAX_BATCH_SIZE",
    "SECURITY_MODEL_CACHE_DIR",
    "SECURITY_USE_FP16",
    # Output
    "LOG_LEVEL",
    "SHOW_RETRIEVAL_LOG",
    "VERBOSE_OUTPUT",
]
