"""Hardcoded constants for the code RAG agent project.

This module contains all constants that are not configurable via environment variables.
"""

from typing import List, Set

# ============================================================================
# REPOSITORY SETTINGS
# ============================================================================

# Patterns to exclude from indexing
REPO_EXCLUDE_PATTERNS: List[str] = [
    "__pycache__/*",
    ".git/*",
    "docs/img/*",  # Exclude images from docs
    "docs/css/*",  # Exclude CSS files
    "docs/overrides/*",  # Exclude MkDocs overrides
    "docs/CNAME",  # Exclude CNAME file
    "scripts/build",  # Exclude build script
    "scripts/clean",  # Exclude clean script
    "scripts/publish",  # Exclude publish script
    "scripts/sync-version",  # Exclude version sync script
    "*.pyc",  # Exclude compiled Python files
    "*.pyo",  # Exclude optimized Python files
    "*.egg-info/*",  # Exclude egg info directories
    ".pytest_cache/*",  # Exclude pytest cache
    ".coverage",  # Exclude coverage files
    "htmlcov/*",  # Exclude HTML coverage reports
]

# ============================================================================
# VECTOR STORE SETTINGS
# ============================================================================

# Vector database collection name
VECTOR_DB_COLLECTION_NAME = "code_chunks"

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

# Common chunking parameters
DEFAULT_MAX_CHUNK_SIZE = 2000  # characters
DEFAULT_CHUNK_OVERLAP = 200    # characters/tokens
DEFAULT_MAX_CHUNK_TOKENS = 1000  # tokens

# Common file extensions to process
TEXT_EXTENSIONS: Set[str] = {'.toml', '.yml', '.yaml', '.txt', '.ini', '.cfg', '.conf', '.json', '.pyi'}
CODE_EXTENSIONS: Set[str] = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.php', '.rb', '.go', '.rs'}

# Markdown processing configuration
MAX_CHUNK_SIZE_MARKDOWN = 1000  # tokens for markdown processing
CHUNK_OVERLAP_MARKDOWN = 200    # tokens for markdown recursive splits

# ============================================================================
# CHUNK SCHEMA SETTINGS
# ============================================================================

# Source types for unified chunk schema
SOURCE_TYPES: List[str] = ["code", "markdown", "text"]

# Valid chunk types for each source type
CHUNK_TYPE_MAP = {
    "code": ["function", "class"],
    "markdown": ["markdown_section", "markdown_section_chunk"],
    "text": ["text_file", "text_file_chunk"]
}
