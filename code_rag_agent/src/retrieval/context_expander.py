"""Context expansion for retrieved chunks.

This module enriches retrieved chunks with parent/sibling/import context to provide
complete understanding without requiring additional retrieval. Different strategies
are used for code, markdown, and text chunks.

v1.2 Optimizations:
- Pre-built lookup dicts for O(1) access instead of O(N) linear scans
- Indexed by (file_path, parent_context) for method siblings
- Indexed by (file_path, class_name) for parent class lookup

Author: Hay Hoffman
Version: 1.2
"""

import logging
from collections import defaultdict

from models.chunk import BaseChunk, CodeChunk, MarkdownChunk, TextChunk
from models.retrieval import ExpandedChunk, RetrievedChunk
from settings import MAX_CHILDREN_SECTIONS, MAX_IMPORTS, MAX_RELATED_METHODS
from src.retrieval.chunk_loader import ChunkLoader
from src.retrieval.metadata_filter import ChunkMetadata

logger = logging.getLogger(__name__)

__all__ = ["ContextExpander"]


class ContextExpander:
    """Expand retrieved chunks with contextual information (v1.2 optimized).

    Enriches chunks based on their source type:
    - Code: Parent class, sibling methods, imports
    - Markdown: Parent headers, child sections
    - Text: No expansion (self-contained)

    v1.2 Optimizations:
    - Pre-built lookup dicts for O(1) parent/sibling lookups
    - No more O(N) linear scans per expansion

    Attributes:
        metadata_index: Mapping of source_type -> list of chunk metadata
        chunk_loader: Chunk loader for fetching related chunks
        _class_lookup: (file_path, class_name) -> ChunkMetadata for parent lookup
        _methods_by_class: (file_path, parent_context) -> list[ChunkMetadata] for siblings
        _markdown_children: (file_path, parent_id) -> list[ChunkMetadata] for children
    """

    def __init__(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        chunk_loader: ChunkLoader
    ):
        """Initialize context expander with pre-built lookup structures.

        Args:
            metadata_index: Metadata index for finding related chunks
            chunk_loader: Chunk loader for loading full chunks
        """
        self.metadata_index = metadata_index
        self.chunk_loader = chunk_loader

        # v1.2: Pre-build lookup structures for O(1) access
        self._class_lookup: dict[tuple[str, str], ChunkMetadata] = {}
        self._methods_by_class: dict[tuple[str, str], list[ChunkMetadata]] = defaultdict(list)
        self._markdown_by_file: dict[str, list[ChunkMetadata]] = defaultdict(list)

        self._build_lookup_structures()

    def _build_lookup_structures(self):
        """Build pre-computed lookup dicts for O(1) access (v1.2).

        Builds:
        - _class_lookup: (file_path, class_name) -> class ChunkMetadata
        - _methods_by_class: (file_path, parent_context) -> list of method ChunkMetadata
        - _markdown_by_file: file_path -> list of markdown ChunkMetadata (sorted by line)
        """
        # Build code lookups
        if "code" in self.metadata_index:
            for meta in self.metadata_index["code"]:
                if meta.chunk_type == "class":
                    # Class lookup by (file_path, class_name)
                    key = (meta.file_path, meta.name)
                    self._class_lookup[key] = meta

                elif meta.chunk_type == "function" and meta.parent_context:
                    # Methods grouped by (file_path, parent_class)
                    key = (meta.file_path, meta.parent_context)
                    self._methods_by_class[key].append(meta)

        # Build markdown lookups
        if "markdown" in self.metadata_index:
            for meta in self.metadata_index["markdown"]:
                self._markdown_by_file[meta.file_path].append(meta)

            # Sort by start_line for efficient child section lookup
            for file_path in self._markdown_by_file:
                self._markdown_by_file[file_path].sort(
                    key=lambda m: m.start_line if m.start_line else 0
                )

        logger.info(
            f"Built context expansion lookups: "
            f"{len(self._class_lookup)} classes, "
            f"{len(self._methods_by_class)} class->methods groups, "
            f"{len(self._markdown_by_file)} markdown files"
        )

    def expand(self, retrieved_chunk: RetrievedChunk) -> ExpandedChunk | None:
        """Expand a retrieved chunk with context.

        Args:
            retrieved_chunk: Retrieved chunk with base information

        Returns:
            ExpandedChunk with parent/sibling/import context added,
            or None if chunk type is not recognized
        """
        chunk = retrieved_chunk.chunk

        # Dispatch to appropriate expander based on source type
        if chunk.source_type == "code" and isinstance(chunk, CodeChunk):
            return self._expand_code_chunk(retrieved_chunk, chunk)
        elif chunk.source_type == "markdown" and isinstance(chunk, MarkdownChunk):
            return self._expand_markdown_chunk(retrieved_chunk, chunk)
        elif chunk.source_type == "text" and isinstance(chunk, TextChunk):
            return self._expand_text_chunk(retrieved_chunk, chunk)
        else:
            logger.warning(
                f"Unknown chunk type: source_type={chunk.source_type}, "
                f"type={type(chunk).__name__}"
            )
            return None

    def _expand_code_chunk(
        self,
        retrieved_chunk: RetrievedChunk,
        chunk: CodeChunk
    ) -> ExpandedChunk:
        """Expand code chunk with parent class, siblings, and imports (v1.2 optimized).

        Uses pre-built lookup dicts for O(1) access.

        Args:
            retrieved_chunk: Retrieved chunk
            chunk: Code chunk to expand

        Returns:
            ExpandedChunk with code context
        """
        parent_chunk: BaseChunk | None = None
        sibling_chunks: list[BaseChunk] = []
        import_chunks: list[BaseChunk] = []

        # 1. Find parent class using lookup dict (O(1))
        if chunk.parent_context and chunk.chunk_type == "function":
            parent_chunk = self._find_parent_class_optimized(chunk)
            if parent_chunk:
                logger.debug(
                    f"Found parent class '{parent_chunk.name}' for method '{chunk.name}'"
                )

        # 2. Find sibling methods using lookup dict (O(1) lookup + O(siblings))
        if chunk.parent_context:
            siblings = self._find_sibling_methods_optimized(chunk)
            sibling_chunks = list(siblings)
            logger.debug(
                f"Found {len(sibling_chunks)} sibling methods for '{chunk.name}'"
            )

        # 3. Find imported chunks (not implemented - would require import resolution)
        if chunk.imports:
            logger.debug(
                f"Chunk '{chunk.name}' has {len(chunk.imports)} imports "
                f"(import resolution not implemented)"
            )

        return ExpandedChunk(
            base_chunk=chunk,
            rrf_score=retrieved_chunk.rrf_score,
            parent_chunk=parent_chunk,
            sibling_chunks=sibling_chunks[:MAX_RELATED_METHODS],
            import_chunks=import_chunks[:MAX_IMPORTS],
            expansion_metadata={
                "expanded": True,
                "parent_added": parent_chunk is not None,
                "siblings_added": len(sibling_chunks[:MAX_RELATED_METHODS]),
                "imports_added": len(import_chunks[:MAX_IMPORTS])
            }
        )

    def _expand_markdown_chunk(
        self,
        retrieved_chunk: RetrievedChunk,
        chunk: MarkdownChunk
    ) -> ExpandedChunk:
        """Expand markdown chunk with child sections (v1.2 optimized).

        Uses pre-built lookup dicts for faster child lookup.

        Args:
            retrieved_chunk: Retrieved chunk
            chunk: Markdown chunk to expand

        Returns:
            ExpandedChunk with markdown context
        """
        child_sections: list[BaseChunk] = []

        # Find child sections using pre-sorted file index
        if chunk.heading_level is not None:
            children = self._find_child_sections_optimized(chunk)
            child_sections = list(children)
            logger.debug(
                f"Found {len(child_sections)} child sections for '{chunk.name}'"
            )

        return ExpandedChunk(
            base_chunk=chunk,
            rrf_score=retrieved_chunk.rrf_score,
            parent_chunk=None,
            child_sections=child_sections[:MAX_CHILDREN_SECTIONS],
            expansion_metadata={
                "expanded": True,
                "children_added": len(child_sections[:MAX_CHILDREN_SECTIONS])
            }
        )

    def _expand_text_chunk(
        self,
        retrieved_chunk: RetrievedChunk,
        chunk: TextChunk
    ) -> ExpandedChunk:
        """Text chunks are self-contained - no expansion needed.

        Args:
            retrieved_chunk: Retrieved chunk
            chunk: Text chunk (no expansion)

        Returns:
            ExpandedChunk with no additional context
        """
        return ExpandedChunk(
            base_chunk=chunk,
            rrf_score=retrieved_chunk.rrf_score,
            parent_chunk=None,
            expansion_metadata={
                "expanded": False,
                "reason": "Text chunks are self-contained"
            }
        )

    def _find_parent_class_optimized(self, chunk: CodeChunk) -> CodeChunk | None:
        """Find parent class using pre-built lookup dict (v1.2 optimized).

        O(1) lookup instead of O(N) scan.

        Args:
            chunk: Code chunk (method) to find parent for

        Returns:
            Parent class chunk or None if not found
        """
        key = (chunk.file_path, chunk.parent_context)
        meta = self._class_lookup.get(key)

        if meta:
            try:
                parent = self.chunk_loader.load_chunk(meta.id)
                if isinstance(parent, CodeChunk):
                    return parent
            except Exception as e:
                logger.error(f"Failed to load parent class {meta.id}: {e}")

        return None

    def _find_sibling_methods_optimized(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Find sibling methods using pre-built lookup dict (v1.2 optimized).

        O(1) lookup + O(siblings) iteration instead of O(N) scan.

        Args:
            chunk: Code chunk to find siblings for

        Returns:
            list of sibling method chunks
        """
        siblings: list[CodeChunk] = []

        key = (chunk.file_path, chunk.parent_context)
        sibling_metas = self._methods_by_class.get(key, [])

        for meta in sibling_metas:
            if meta.id != chunk.id:  # Exclude self
                try:
                    sibling = self.chunk_loader.load_chunk(meta.id)
                    if isinstance(sibling, CodeChunk):
                        siblings.append(sibling)
                except Exception as e:
                    logger.error(f"Failed to load sibling method {meta.id}: {e}")

        return siblings

    def _find_child_sections_optimized(self, chunk: MarkdownChunk) -> list[MarkdownChunk]:
        """Find child sections using pre-sorted file index (v1.2 optimized).

        Uses pre-sorted markdown index for more efficient lookup.

        Args:
            chunk: Markdown chunk to find children for

        Returns:
            list of child section chunks
        """
        children: list[MarkdownChunk] = []

        if chunk.heading_level is None or chunk.start_line is None or chunk.end_line is None:
            return children

        # Get pre-sorted markdown sections for this file
        file_sections = self._markdown_by_file.get(chunk.file_path, [])

        for meta in file_sections:
            # Child must have deeper heading level and be within parent's line range
            if (meta.heading_level is not None and
                meta.heading_level > chunk.heading_level and
                meta.start_line is not None and
                meta.end_line is not None and
                meta.start_line > chunk.start_line and
                meta.end_line <= chunk.end_line):
                try:
                    child = self.chunk_loader.load_chunk(meta.id)
                    if isinstance(child, MarkdownChunk):
                        children.append(child)
                except Exception as e:
                    logger.error(f"Failed to load child section {meta.id}: {e}")

        return children

    # Keep original methods for backward compatibility (deprecated)
    def _find_parent_class(self, chunk: CodeChunk) -> CodeChunk | None:
        """Find parent class by name in same file (deprecated - use _find_parent_class_optimized)."""
        return self._find_parent_class_optimized(chunk)

    def _find_sibling_methods(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Find other methods in same class (deprecated - use _find_sibling_methods_optimized)."""
        return self._find_sibling_methods_optimized(chunk)

    def _find_child_sections(self, chunk: MarkdownChunk) -> list[MarkdownChunk]:
        """Find child sections (deprecated - use _find_child_sections_optimized)."""
        return self._find_child_sections_optimized(chunk)

    def expand_batch(
        self,
        retrieved_chunks: list[RetrievedChunk]
    ) -> list[ExpandedChunk]:
        """Expand multiple retrieved chunks.

        Args:
            retrieved_chunks: List of retrieved chunks to expand

        Returns:
            List of expanded chunks with context (None results filtered out)
        """
        expanded = []
        for chunk in retrieved_chunks:
            result = self.expand(chunk)
            if result is not None:
                expanded.append(result)
        return expanded
