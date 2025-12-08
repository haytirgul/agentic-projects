"""Chunk loader for loading full chunks from disk.

This module provides efficient chunk loading from JSON files with caching
to minimize disk I/O during retrieval.

Author: Hay Hoffman
"""

import json
import logging
from pathlib import Path

from models.chunk import BaseChunk, create_chunk_from_dict

logger = logging.getLogger(__name__)

__all__ = ["ChunkLoader"]


class ChunkLoader:
    """Load full chunks from disk with in-memory caching.

    Chunks are stored in JSON files organized by source_type. The loader
    maintains an in-memory cache (chunk_id -> chunk) to avoid repeated
    disk reads during retrieval.

    Attributes:
        chunks_dir: Directory containing chunk JSON files
        chunk_cache: In-memory cache of loaded chunks
        chunks_by_source: Mapping of source_type -> list of chunks
    """

    def __init__(self, chunks_dir: Path):
        """Initialize chunk loader.

        Args:
            chunks_dir: Directory containing chunk JSON files
                        (e.g., data/processed/)
        """
        self.chunks_dir = Path(chunks_dir)
        self.chunk_cache: dict[str, BaseChunk] = {}
        self.chunks_by_source: dict[str, list] = {}

        # Load all chunks on initialization
        self._load_all_chunks()

    def _load_all_chunks(self):
        """Load all chunks from JSON files into memory.

        Loads chunks from:
        - code_chunks.json
        - markdown_chunks.json
        - text_chunks.json
        - all_chunks.json (fallback)
        """
        # Try loading from combined file first
        all_chunks_file = self.chunks_dir / "all_chunks.json"
        if all_chunks_file.exists():
            logger.info(f"Loading chunks from {all_chunks_file}")
            self._load_chunks_from_file(all_chunks_file)
            return

        # Load from individual files
        for source_type in ["code", "markdown", "text"]:
            chunks_file = self.chunks_dir / f"{source_type}_chunks.json"
            if chunks_file.exists():
                logger.info(f"Loading chunks from {chunks_file}")
                self._load_chunks_from_file(chunks_file, source_type)

    def _load_chunks_from_file(
        self,
        file_path: Path,
        source_type: str | None = None
    ):
        """Load chunks from a JSON file.

        Args:
            file_path: Path to JSON file containing chunks
            source_type: Optional source type filter
        """
        try:
            with open(file_path, encoding='utf-8') as f:
                chunks_data = json.load(f)

            loaded_count = 0
            for chunk_dict in chunks_data:
                # Create chunk from dict using factory function
                chunk = create_chunk_from_dict(chunk_dict)

                # Add to cache
                self.chunk_cache[chunk.id] = chunk

                # Add to source type index
                chunk_source = chunk.source_type
                if chunk_source not in self.chunks_by_source:
                    self.chunks_by_source[chunk_source] = []
                self.chunks_by_source[chunk_source].append(chunk)

                loaded_count += 1

            logger.info(
                f"Loaded {loaded_count} chunks from {file_path.name}"
            )

        except Exception as e:
            logger.error(f"Failed to load chunks from {file_path}: {e}")
            raise

    def load_chunk(self, chunk_id: str) -> BaseChunk:
        """Load a single chunk by ID.

        Args:
            chunk_id: Unique chunk identifier (MD5 hash)

        Returns:
            BaseChunk instance (CodeChunk, MarkdownChunk, or TextChunk)

        Raises:
            KeyError: If chunk_id not found
        """
        if chunk_id not in self.chunk_cache:
            raise KeyError(
                f"Chunk {chunk_id} not found in cache. "
                f"Available chunks: {len(self.chunk_cache)}"
            )

        return self.chunk_cache[chunk_id]

