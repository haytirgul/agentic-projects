"""Context expansion for retrieved chunks.

This module enriches retrieved chunks with parent/sibling/import context to provide
complete understanding without requiring additional retrieval. Different strategies
are used for code, markdown, and text chunks.

Author: Hay Hoffman
Version: 1.1
"""

import logging

from settings import MAX_CHILDREN_SECTIONS, MAX_IMPORTS, MAX_RELATED_METHODS
from models.chunk import BaseChunk, CodeChunk, MarkdownChunk, TextChunk
from models.retrieval import ExpandedChunk, RetrievedChunk

from src.retrieval.chunk_loader import ChunkLoader
from src.retrieval.metadata_filter import ChunkMetadata

logger = logging.getLogger(__name__)

__all__ = ["ContextExpander"]


class ContextExpander:
    """Expand retrieved chunks with contextual information.

    Enriches chunks based on their source type:
    - Code: Parent class, sibling methods, imports
    - Markdown: Parent headers, child sections
    - Text: No expansion (self-contained)

    Attributes:
        metadata_index: Mapping of source_type -> list of chunk metadata
        chunk_loader: Chunk loader for fetching related chunks
    """

    def __init__(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        chunk_loader: ChunkLoader
    ):
        """Initialize context expander.

        Args:
            metadata_index: Metadata index for finding related chunks
            chunk_loader: Chunk loader for loading full chunks
        """
        self.metadata_index = metadata_index
        self.chunk_loader = chunk_loader

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
        """Expand code chunk with parent class, siblings, and imports.

        Context hierarchy:
        1. Parent class (if method)
        2. Related methods (same class, up to MAX_RELATED_METHODS)
        3. Import statements (up to MAX_IMPORTS)

        Args:
            retrieved_chunk: Retrieved chunk
            chunk: Code chunk to expand

        Returns:
            ExpandedChunk with code context
        """
        parent_chunk: BaseChunk | None = None
        sibling_chunks: list[BaseChunk] = []
        import_chunks: list[BaseChunk] = []

        # 1. Find parent class (if this is a method)
        if chunk.parent_context and chunk.chunk_type == "function":
            parent_chunk = self._find_parent_class(chunk)
            if parent_chunk:
                logger.debug(
                    f"Found parent class '{parent_chunk.name}' for method '{chunk.name}'"
                )

        # 2. Find sibling methods (other methods in same class)
        if chunk.parent_context:
            siblings = self._find_sibling_methods(chunk)
            sibling_chunks = list(siblings)  # Cast to list[BaseChunk]
            logger.debug(
                f"Found {len(sibling_chunks)} sibling methods for '{chunk.name}'"
            )

        # 3. Find imported chunks (not implemented - would require import resolution)
        # For now, we just log the imports that are present
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
        """Expand markdown chunk with child sections.

        Context hierarchy:
        1. Child sections (up to MAX_CHILDREN_SECTIONS)

        Note: Parent headers are already in chunk.headers dict

        Args:
            retrieved_chunk: Retrieved chunk
            chunk: Markdown chunk to expand

        Returns:
            ExpandedChunk with markdown context
        """
        child_sections: list[BaseChunk] = []

        # Find child sections (subsections under this header)
        if chunk.heading_level is not None:
            children = self._find_child_sections(chunk)
            child_sections = list(children)  # Cast to list[BaseChunk]
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

    def _find_parent_class(self, chunk: CodeChunk) -> CodeChunk | None:
        """Find parent class by name in same file.

        Args:
            chunk: Code chunk (method) to find parent for

        Returns:
            Parent class chunk or None if not found
        """
        if "code" not in self.metadata_index:
            return None

        for meta in self.metadata_index["code"]:
            if (meta.chunk_type == "class" and
                meta.name == chunk.parent_context and
                meta.file_path == chunk.file_path):
                try:
                    parent = self.chunk_loader.load_chunk(meta.id)
                    if isinstance(parent, CodeChunk):
                        return parent
                except Exception as e:
                    logger.error(f"Failed to load parent class {meta.id}: {e}")

        return None

    def _find_sibling_methods(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Find other methods in same class.

        Args:
            chunk: Code chunk to find siblings for

        Returns:
            list of sibling method chunks
        """
        siblings: list[CodeChunk] = []

        if "code" not in self.metadata_index:
            return siblings

        for meta in self.metadata_index["code"]:
            if (meta.chunk_type == "function" and
                meta.parent_context == chunk.parent_context and
                meta.file_path == chunk.file_path and
                meta.id != chunk.id):  # Exclude self
                try:
                    sibling = self.chunk_loader.load_chunk(meta.id)
                    if isinstance(sibling, CodeChunk):
                        siblings.append(sibling)
                except Exception as e:
                    logger.error(f"Failed to load sibling method {meta.id}: {e}")

        return siblings

    def _find_child_sections(self, chunk: MarkdownChunk) -> list[MarkdownChunk]:
        """Find child sections (deeper heading levels in same file).

        Args:
            chunk: Markdown chunk to find children for

        Returns:
            list of child section chunks
        """
        children: list[MarkdownChunk] = []

        if "markdown" not in self.metadata_index:
            return children

        if chunk.heading_level is None:
            return children

        for meta in self.metadata_index["markdown"]:
            # Child must be in same file and have deeper heading level
            if (meta.file_path == chunk.file_path and
                meta.heading_level is not None and
                meta.heading_level > chunk.heading_level and
                meta.start_line is not None and
                meta.end_line is not None and
                chunk.start_line is not None and
                chunk.end_line is not None and
                meta.start_line > chunk.start_line and
                meta.end_line < chunk.end_line):
                try:
                    child = self.chunk_loader.load_chunk(meta.id)
                    if isinstance(child, MarkdownChunk):
                        children.append(child)
                except Exception as e:
                    logger.error(f"Failed to load child section {meta.id}: {e}")

        return children

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
