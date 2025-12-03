import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# RAG Settings
CHROMA_PERSIST_DIRECTORY = str(DATA_DIR / "vector_db")
COLLECTION_NAME = "documentation_embeddings"

# Embedding Model: Google Gemini or Hugging Face (local)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Sentence Transformers cache directory
SENTENCE_TRANSFORMERS_HOME = os.getenv("SENTENCE_TRANSFORMERS_HOME", str(DATA_DIR / "sentence_transformers_cache"))
os.environ["SENTENCE_TRANSFORMERS_HOME"] = SENTENCE_TRANSFORMERS_HOME

# Agent Mode: "offline" (default) or "online" (with web search)
AGENT_MODE = os.getenv("AGENT_MODE", "offline")

# Validate AGENT_MODE
if AGENT_MODE not in ["offline", "online"]:
    raise ValueError(f"Invalid AGENT_MODE: {AGENT_MODE}. Must be 'offline' or 'online'")

# RAG Scoring Weights
if AGENT_MODE == "online":
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.15"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.15"))
    WEB_BONUS = float(os.getenv("WEB_BONUS", "0.7"))
else:
    VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT_OFFLINE", "0.7"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT_OFFLINE", "0.3"))
    WEB_BONUS = 0.0

# Security Configuration
SECURITY_ENABLED = os.getenv("SECURITY_ENABLED", "false").lower() == "true"
SECURITY_MODEL_CACHE_DIR = os.getenv("SECURITY_MODEL_CACHE_DIR", None)
SECURITY_FAIL_CLOSED = os.getenv("SECURITY_FAIL_CLOSED", "true").lower() == "true"
SECURITY_MAX_BATCH_SIZE = int(os.getenv("SECURITY_MAX_BATCH_SIZE", "128"))
SECURITY_USE_FP16 = os.getenv("SECURITY_USE_FP16", "false").lower() == "true"

# LLM Settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not configured. Please set it in your .env file."
    )

# LangSmith Observability (Optional)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-agent")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# Model Tiers: fast, intermediate, slow
MODEL_FAST = os.getenv("MODEL_FAST", "gemini-2.5-flash-lite")
MODEL_INTERMEDIATE = os.getenv("MODEL_INTERMEDIATE", "gemini-2.5-flash")
MODEL_SLOW = os.getenv("MODEL_SLOW", "gemini-2.5-pro")

LEVEL_TO_MODEL = {
    "fast": MODEL_FAST,
    "intermediate": MODEL_INTERMEDIATE,
    "slow": MODEL_SLOW,
}

# Intent Classification Settings
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# LLM Parameters
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# Streaming & Optimization
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
LAZY_INITIALIZATION = os.getenv("LAZY_INITIALIZATION", "true").lower() == "true"

# Web Search (Online Mode)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))

# Validate online mode
if AGENT_MODE == "online":
    if not TAVILY_API_KEY:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Online mode enabled but TAVILY_API_KEY missing. Falling back to offline.")
        AGENT_MODE = "offline"
    else:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Agent running in ONLINE mode")
else:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Agent running in OFFLINE mode")

# Legacy settings for backward compatibility
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192")) if os.getenv("LLM_MAX_TOKENS") else None

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


