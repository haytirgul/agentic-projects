"""Process configuration and text files into chunks.

This script processes configuration files (TOML, YAML, INI) and other text files
that are included in the RAG system but not handled by specialized processors.

Usage:
    python scripts/data_pipeline/processing/process_text_files.py
"""

import logging
import sys
from pathlib import Path
from typing import Any

# Setup project paths first - must be done before importing from const/settings/models
from utils import setup_project_paths

project_root = setup_project_paths()

from settings import (
    HTTPX_REPO_DIR,
    MAX_CHUNK_SIZE_TEXT,
    REPO_EXCLUDE_PATTERNS,
    TEXT_CHUNKS_FILE,
    TEXT_EXTENSIONS,
)
from utils import (
    create_chunk_id,
    estimate_token_count,
    generate_processing_stats,
    print_processing_stats,
    safe_read_file,
    save_json_chunks,
    should_exclude,
)

logger = logging.getLogger(__name__)

from models.chunk import TextChunk


def chunk_text_content(content: str, file_path: Path) -> list[dict[str, Any]]:
    """Chunk text content based on size or structure with unified schema."""
    chunks = []

    # Handle file path for production and testing
    try:
        relative_path = str(file_path.relative_to(HTTPX_REPO_DIR))
    except ValueError:
        relative_path = str(file_path)

    # For small files, create single chunk
    if len(content) <= MAX_CHUNK_SIZE_TEXT:
        chunk_id = create_chunk_id(file_path, 1, content)
        token_count = estimate_token_count(content)

        try:
            # Create and validate chunk with Pydantic
            text_chunk = TextChunk(
                id=chunk_id,
                source_type="text",
                chunk_type="text_file",
                content=content,
                token_count=token_count,
                file_path=relative_path,
                filename=file_path.name,
                start_line=1,
                end_line=len(content.split('\n')),
                name=file_path.name,
                full_name=file_path.name,
                parent_context="",
                file_extension=file_path.suffix
            )
            chunks.append(text_chunk.model_dump(exclude_none=True))
        except Exception as e:
            logger.error(
                f"Pydantic validation failed for text file {file_path}. "
                f"Error: {e}. Skipping chunk - requires human examination."
            )
            # Skip invalid chunk
    else:
        # For large files, split into chunks
        lines = content.split('\n')
        current_chunk_lines = []
        current_chunk_start = 1

        for i, line in enumerate(lines, 1):
            current_chunk_lines.append(line)

            # Check if chunk is getting too large
            chunk_content = '\n'.join(current_chunk_lines)
            if len(chunk_content) >= MAX_CHUNK_SIZE_TEXT or i == len(lines):
                chunk_id = create_chunk_id(file_path, current_chunk_start, chunk_content)
                token_count = estimate_token_count(chunk_content)

                try:
                    # Create and validate chunk with Pydantic
                    text_chunk = TextChunk(
                        id=chunk_id,
                        source_type="text",
                        chunk_type="text_file_chunk",
                        content=chunk_content,
                        token_count=token_count,
                        file_path=relative_path,
                        filename=file_path.name,
                        start_line=current_chunk_start,
                        end_line=i,
                        name=f"{file_path.name} (part {len(chunks) + 1})",
                        full_name=f"{file_path.name} (part {len(chunks) + 1})",
                        parent_context=file_path.name,
                        file_extension=file_path.suffix
                    )
                    chunks.append(text_chunk.model_dump(exclude_none=True))
                except Exception as e:
                    logger.error(
                        f"Pydantic validation failed for text file chunk "
                        f"in {file_path}:{current_chunk_start}-{i}. "
                        f"Error: {e}. Skipping chunk - requires human examination."
                    )
                    # Skip invalid chunk

                # Reset for next chunk
                current_chunk_lines = []
                current_chunk_start = i + 1

    return chunks


def process_text_file(file_path: Path) -> list[dict[str, Any]]:
    """Process a text file into chunks."""
    chunks = []

    content = safe_read_file(file_path)
    if content and content.strip():  # Only process non-empty files
        chunks = chunk_text_content(content, file_path)
    elif content is None:
        logger.warning(f"Cannot read {file_path}, skipping")

    return chunks


def main() -> int:
    """Main entry point.

    Returns:
        0 on success, 1 on failure
    """
    logger.info("Starting text file processing...")

    if not HTTPX_REPO_DIR.exists():
        logger.error(f"Repository not found at {HTTPX_REPO_DIR}")
        logger.error("Run clone_httpx_repo.py first")
        return 1

    # Find all text files to process
    text_files = []
    for ext in TEXT_EXTENSIONS:
        text_files.extend(list(HTTPX_REPO_DIR.rglob(f"*{ext}")))

    # Remove duplicates
    text_files = list(set(text_files))
    logger.info(f"Found {len(text_files)} text files")

    # Filter excluded files
    included_files = [f for f in text_files if not should_exclude(f, HTTPX_REPO_DIR, REPO_EXCLUDE_PATTERNS)]
    logger.info(f"Processing {len(included_files)} files (excluded {len(text_files) - len(included_files)})")

    # Process files
    all_chunks = []
    for i, file_path in enumerate(included_files, 1):
        if i % 10 == 0:
            logger.info(f"Processing file {i}/{len(included_files)}: {file_path.name}")

        chunks = process_text_file(file_path)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} text chunks")

    # Save chunks to JSON using utility function
    if save_json_chunks(all_chunks, TEXT_CHUNKS_FILE):
        logger.info(f"[SUCCESS] Saved chunks to {TEXT_CHUNKS_FILE}")
    else:
        logger.error("[ERROR] Failed to save chunks")
        return 1

    # Generate and print statistics using utility functions
    stats = generate_processing_stats(all_chunks)
    print_processing_stats(stats, logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
