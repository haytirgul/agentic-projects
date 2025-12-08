"""Shared utilities for data processing pipeline.

This module contains common functionality used across all processing scripts,
including file filtering, logging setup, path management, and utility functions.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "setup_project_paths",
    "should_exclude",
    "estimate_token_count",
    "calculate_chunk_size",
    "safe_read_file",
    "save_json_chunks",
    "create_chunk_id",
    "validate_chunk",
    "generate_processing_stats",
    "print_processing_stats",
]

# ============================================================================
# PROJECT SETUP
# ============================================================================

def setup_project_paths() -> Path:
    """Setup project paths and add to sys.path for imports.

    This function adds the project root to sys.path so that modules can
    import from settings, const, etc.

    Returns:
        Path to project root
    """
    # Go up from scripts/data_pipeline/processing to project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Add to sys.path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root

# ============================================================================
# FILE FILTERING AND EXCLUSION
# ============================================================================

def should_exclude(file_path: Path, repo_root: Path, exclude_patterns: list[str]) -> bool:
    """Check if file should be excluded based on patterns.

    Args:
        file_path: Path to the file
        repo_root: Root of the repository
        exclude_patterns: list of patterns to exclude

    Returns:
        True if file should be excluded
    """
    try:
        relative_path = file_path.relative_to(repo_root)
        relative_str = str(relative_path)

        for pattern in exclude_patterns:
            # Simple pattern matching (not full glob)
            if pattern.startswith("*"):
                if relative_str.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("/*"):
                if relative_str.startswith(pattern[:-2]):
                    return True
            elif pattern in relative_str:
                return True

        return False
    except ValueError:
        return True


# ============================================================================
# TOKEN AND SIZE UTILITIES
# ============================================================================

def estimate_token_count(text: str) -> int:
    """Rough token count estimation (words / 0.75).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return max(1, len(text.split()) // 3 * 4)  # Conservative estimate


def calculate_chunk_size(content: str, max_size: int = 2000) -> int:
    """Calculate appropriate chunk size based on content.

    Args:
        content: Text content
        max_size: Maximum chunk size in characters (default: 2000)

    Returns:
        Recommended chunk size
    """
    content_length = len(content)

    if content_length <= max_size:
        return content_length

    # For large content, aim for chunks around max_size
    return min(max_size, content_length // 2 + 1)


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def safe_read_file(file_path: Path, encoding: str = 'utf-8') -> str | None:
    """Safely read a file with error handling.

    Args:
        file_path: Path to file
        encoding: File encoding

    Returns:
        File content or None if error
    """
    try:
        with open(file_path, encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding or return None
        try:
            with open(file_path, encoding='latin-1') as f:
                return f.read()
        except Exception:
            return None
    except Exception:
        return None


def save_json_chunks(chunks: list[dict[str, Any]], output_file: Path) -> bool:
    """Save chunks to JSON file with error handling.

    Args:
        chunks: list of chunk dictionaries
        output_file: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving chunks to {output_file}: {e}")
        return False


# ============================================================================
# CHUNKING UTILITIES
# ============================================================================

def create_chunk_id(file_path: Path, start_line: int, content_preview: str, max_length: int = 16) -> str:
    """Create a unique chunk ID.

    Args:
        file_path: File path
        start_line: Starting line number
        content_preview: Preview of content
        max_length: Maximum ID length

    Returns:
        Unique chunk identifier
    """
    import hashlib
    content_hash = hashlib.md5(
        f"{file_path}:{start_line}:{content_preview[:100]}".encode()
    ).hexdigest()
    return content_hash[:max_length]


def validate_chunk(chunk: dict[str, Any]) -> bool:
    """Validate chunk structure.

    Args:
        chunk: Chunk dictionary

    Returns:
        True if chunk is valid
    """
    required_fields = ['file_path', 'start_line', 'end_line', 'chunk_type', 'content']
    return all(field in chunk for field in required_fields)


# ============================================================================
# STATISTICS AND REPORTING
# ============================================================================

def generate_processing_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate comprehensive statistics about processed chunks.

    Args:
        chunks: list of chunk dictionaries

    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {}

    chunk_types: dict[str, int] = {}
    file_types: dict[str, int] = {}

    # Chunk types
    for chunk in chunks:
        ctype = chunk.get('chunk_type', 'unknown')
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1

    # File types
    for chunk in chunks:
        file_path = chunk.get('file_path', '')
        if '.' in file_path:
            ext = Path(file_path).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

    stats = {
        'total_chunks': len(chunks),
        'chunk_types': chunk_types,
        'file_types': file_types,
        'token_stats': {},
        'size_stats': {}
    }

    # Token statistics
    token_counts = [chunk.get('token_count', 0) for chunk in chunks if 'token_count' in chunk]
    if token_counts:
        stats['token_stats'] = {
            'average': sum(token_counts) / len(token_counts),
            'max': max(token_counts),
            'min': min(token_counts),
            'total': sum(token_counts)
        }

    # Size statistics
    sizes = [len(chunk.get('content', '')) for chunk in chunks]
    if sizes:
        stats['size_stats'] = {
            'average_chars': sum(sizes) / len(sizes),
            'max_chars': max(sizes),
            'min_chars': min(sizes),
            'total_chars': sum(sizes)
        }

    return stats


def print_processing_stats(stats: dict[str, Any], logger: logging.Logger | None = None):
    """Print formatted processing statistics.

    Args:
        stats: Statistics dictionary from generate_processing_stats
        logger: Optional logger (defaults to print)
    """
    output = logger.info if logger else print

    if not stats:
        output("No statistics available")
        return

    output(f"📊 Total chunks: {stats.get('total_chunks', 0)}")

    # Chunk types
    if 'chunk_types' in stats:
        output("Chunk types:")
        for ctype, count in sorted(stats['chunk_types'].items()):
            output(f"  {ctype}: {count}")

    # File types
    if 'file_types' in stats:
        output("File types:")
        for ext, count in sorted(stats['file_types'].items()):
            output(f"  {ext}: {count}")

    # Token stats
    if 'token_stats' in stats:
        ts = stats['token_stats']
        output("Token statistics:")
        output(f"  Average tokens per chunk: {ts.get('average', 0):.1f}")
        output(f"  Max tokens: {ts.get('max', 0)}")
        output(f"  Min tokens: {ts.get('min', 0)}")
        output(f"  Total tokens: {ts.get('total', 0)}")

    # Size stats
    if 'size_stats' in stats:
        ss = stats['size_stats']
        output("Size statistics:")
        output(f"  Average characters: {ss.get('average_chars', 0):.1f}")
        output(f"  Max characters: {ss.get('max_chars', 0)}")
        output(f"  Total characters: {ss.get('total_chars', 0)}")

