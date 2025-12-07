"""Metadata filtering for retrieval requests.

This module implements RAM-based metadata filtering to reduce the search space
before running BM25 and FAISS searches. Filters by source_type and file patterns.

Note: Folder filtering uses SOFT matching (boost scores, not exclusion) to prevent
LLM folder inference from reducing recall. See HybridRetriever for folder boost logic.

Author: Hay Hoffman
Version: 1.2
"""

import logging
from fnmatch import fnmatch

from models.retrieval import RetrievalRequest

logger = logging.getLogger(__name__)

__all__ = ["MetadataFilter", "ChunkMetadata"]


class ChunkMetadata:
    """Lightweight metadata for filtering and context expansion (kept in RAM).

    This class contains fields needed for filtering and context expansion,
    avoiding loading full chunk content into memory.

    Attributes:
        id: Unique chunk identifier (MD5 hash)
        source_type: Document category (code, markdown, text)
        filename: Base filename (e.g., "client.py")
        file_path: Relative path from repo root
        chunk_type: Type of chunk (function, class, markdown_section, etc.)
        name: Name of the chunk (function name, class name, heading text)
        parent_context: Parent context for code chunks (class name for methods)
        start_line: Starting line number in file
        end_line: Ending line number in file
        heading_level: Heading level for markdown chunks (1-6)
    """

    def __init__(
        self,
        id: str,
        source_type: str,
        filename: str,
        file_path: str,
        chunk_type: str | None = None,
        name: str | None = None,
        parent_context: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        heading_level: int | None = None,
    ):
        self.id = id
        self.source_type = source_type
        self.filename = filename
        self.file_path = file_path
        self.chunk_type = chunk_type
        self.name = name
        self.parent_context = parent_context
        self.start_line = start_line
        self.end_line = end_line
        self.heading_level = heading_level


class MetadataFilter:
    """Apply metadata filters before search (v1.2 - soft folder filtering).

    Filters chunks by source_type and file patterns to reduce the search space.
    Folder filtering is now SOFT (handled via RRF boost in HybridRetriever).

    v1.2 Change: Folders no longer exclude chunks - they boost RRF scores instead.
    This prevents LLM folder inference from reducing recall when it guesses wrong.

    Filter order:
    1. Source type (select relevant indices)
    2. File patterns (if specified) - HARD filter
    3. Folders - NOT applied here (soft boost in RRF)

    Attributes:
        metadata_index: Mapping of source_type -> list of chunk metadata
        request: Retrieval request with filter criteria
    """

    def __init__(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        request: RetrievalRequest
    ):
        """Initialize metadata filter.

        Args:
            metadata_index: Mapping of source_type -> list of ChunkMetadata
            request: RetrievalRequest with filter criteria
        """
        self.metadata_index = metadata_index
        self.request = request

    def apply(self) -> list[str]:
        """Return filtered chunk IDs that match source_type and file_patterns.

        v1.2: Folder filtering removed (now soft boost in RRF).

        Filter order:
        1. Source type (select relevant indices) - HARD filter
        2. File patterns (if specified) - HARD filter
        3. Folders - NOT applied (soft boost handled in HybridRetriever)

        Returns:
            list of chunk IDs that pass source_type and file_pattern filters
        """
        candidate_ids = []

        # 1. Filter by source_type (multiple types allowed)
        for source_type in self.request.source_types:
            # Get chunks for this source type
            chunks_for_type = self.metadata_index.get(source_type, [])

            for chunk_meta in chunks_for_type:
                # 2. Filter by file patterns (if specified) - HARD filter
                if self.request.file_patterns:
                    if not self._matches_file_pattern(
                        chunk_meta.filename,
                        self.request.file_patterns
                    ):
                        continue

                # 3. Folders - NOT filtered here (soft boost in RRF)
                # This prevents LLM folder inference from reducing recall

                # Passed all filters
                candidate_ids.append(chunk_meta.id)

        logger.debug(
            f"Metadata filter: {len(candidate_ids)} candidates "
            f"from source_types={self.request.source_types}, "
            f"file_patterns={self.request.file_patterns} "
            f"(folders={self.request.folders} will boost RRF scores)"
        )

        return candidate_ids

    def _matches_file_pattern(
        self,
        filename: str,
        patterns: list[str]
    ) -> bool:
        """Check if filename matches any pattern (supports wildcards).

        Args:
            filename: Base filename (e.g., "client.py")
            patterns: list of glob patterns (e.g., ["*.py", "test_*.py"])

        Returns:
            True if filename matches any pattern, False otherwise

        Examples:
            >>> filter._matches_file_pattern("client.py", ["*.py"])
            True
            >>> filter._matches_file_pattern("README.md", ["*.py"])
            False
            >>> filter._matches_file_pattern("test_client.py", ["test_*.py"])
            True
        """
        return any(fnmatch(filename, pattern) for pattern in patterns)

    def _matches_folder(
        self,
        file_path: str,
        folders: list[str]
    ) -> bool:
        """Check if file path starts with any target folder.

        Args:
            file_path: Relative file path (e.g., "src/retrieval/hybrid.py")
            folders: list of folder prefixes (e.g., ["src/retrieval/"])

        Returns:
            True if file_path starts with any folder, False otherwise

        Examples:
            >>> filter._matches_folder("src/retrieval/hybrid.py", ["src/retrieval/"])
            True
            >>> filter._matches_folder("tests/test_hybrid.py", ["src/"])
            False
        """
        return any(file_path.startswith(folder) for folder in folders)
