"""Retrieval models for the Code RAG Agent.

This module defines the data structures for the retrieval pipeline as specified
in RETRIEVAL_ARCHITECTURE.md. All models align with the unified chunk schema
from CHUNKING_ARCHITECTURE.md.

Author: Hay Hoffman
Version: 1.1
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from models.chunk import BaseChunk

__all__ = [
    "RetrievalRequest",
    "RouterOutput",
    "RetrievedChunk",
    "ExpandedChunk"
]


class RetrievalRequest(BaseModel):
    """Single retrieval request from router.

    Represents one piece of information needed to answer the user's query.
    Router may generate multiple requests for complex queries.

    Attributes:
        query: Specific search query string
        source_types: Document types to search (code, markdown, text)
        folders: Target folders (empty = search all)
        file_patterns: File patterns for fine-grained filtering
        reasoning: Why this information is needed for synthesis
    """

    query: str = Field(
        ...,
        min_length=5,
        description="Specific search query"
    )

    source_types: list[Literal["code", "markdown", "text"]] = Field(
        ...,
        min_length=1,
        description="Document types to search (can combine multiple)"
    )

    folders: list[str] = Field(
        default_factory=list,
        description="Target folders (empty = search all). Format: 'src/retrieval/'"
    )

    file_patterns: list[str] = Field(
        default_factory=list,
        description="File patterns. Examples: ['*.md', 'test_*.py', 'README.md']"
    )

    reasoning: str = Field(
        ...,
        min_length=10,
        description="WHY this information is needed (context for synthesis)"
    )

    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """Ensure query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class RouterOutput(BaseModel):
    """Router output with multiple retrieval requests.

    The router analyzes user queries and generates parallel retrieval requests.
    Each request targets a distinct piece of information needed for the answer.

    Attributes:
        cleaned_query: User query cleaned from fluff/filler words
        retrieval_requests: Parallel retrieval requests (max 5 for cost control)
    """

    cleaned_query: str = Field(
        ...,
        description="User query cleaned from fluff"
    )

    retrieval_requests: list[RetrievalRequest] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Parallel retrieval requests (max 5)"
    )

    @field_validator('retrieval_requests')
    @classmethod
    def validate_unique_queries(cls, v: list[RetrievalRequest]) -> list[RetrievalRequest]:
        """Ensure queries are distinct (no duplicate requests)."""
        queries = [req.query.lower() for req in v]
        if len(queries) != len(set(queries)):
            raise ValueError(
                "Duplicate queries detected. Router must unify similar requests."
            )
        return v


class RetrievedChunk(BaseModel):
    """Retrieved chunk with ranking metadata.

    Wraps a BaseChunk with retrieval-specific metadata like RRF score
    and request reasoning.

    Attributes:
        chunk: The actual chunk data (from unified schema)
        rrf_score: Reciprocal Rank Fusion score (0.0-1.0)
        request_reasoning: Why this chunk was retrieved (from RetrievalRequest)
        source_query: The specific query that retrieved this chunk
    """

    chunk: BaseChunk = Field(
        ...,
        description="The retrieved chunk (unified schema)"
    )

    rrf_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Reciprocal Rank Fusion score from hybrid search"
    )

    request_reasoning: str = Field(
        ...,
        description="Why this chunk was retrieved (from RetrievalRequest.reasoning)"
    )

    source_query: str = Field(
        ...,
        description="The specific query that retrieved this chunk"
    )

    @property
    def citation(self) -> str:
        """Generate citation string for this chunk.

        Returns:
            Citation in format: "file_path:start_line-end_line"
        """
        return f"{self.chunk.file_path}:{self.chunk.start_line}-{self.chunk.end_line}"

    @property
    def preview(self) -> str:
        """Get preview of chunk content (first 200 chars).

        Returns:
            Preview string with ellipsis if truncated
        """
        if len(self.chunk.content) <= 200:
            return self.chunk.content
        return self.chunk.content[:200] + "..."


class ExpandedChunk(BaseModel):
    """Chunk with expanded context for synthesis.

    Context expansion enriches chunks with parent/sibling/import information
    to provide complete understanding without requiring additional retrieval.

    Attributes:
        base_chunk: The original retrieved chunk
        rrf_score: Ranking score from retrieval
        parent_chunk: Parent class or module (for methods/functions)
        sibling_chunks: Related methods in same class
        import_chunks: Imported dependencies
        child_sections: Child markdown sections
        expansion_metadata: Details about what was expanded
    """

    base_chunk: BaseChunk = Field(
        ...,
        description="Original retrieved chunk"
    )

    rrf_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="RRF score from retrieval"
    )

    # Context expansion fields
    parent_chunk: BaseChunk | None = Field(
        None,
        description="Parent class/module chunk (for code only)"
    )

    sibling_chunks: list[BaseChunk] = Field(
        default_factory=list,
        description="Related methods in same class (code only)"
    )

    import_chunks: list[BaseChunk] = Field(
        default_factory=list,
        description="Chunks for imported dependencies (code only)"
    )

    child_sections: list[BaseChunk] = Field(
        default_factory=list,
        description="Child markdown sections (markdown only)"
    )

    expansion_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about expansion (parents added, siblings added, etc.)"
    )

    @property
    def total_tokens(self) -> int:
        """Calculate total token count including expanded context.

        Returns:
            Sum of tokens from base chunk and all expanded chunks
        """
        total = self.base_chunk.token_count

        if self.parent_chunk:
            total += self.parent_chunk.token_count

        total += sum(chunk.token_count for chunk in self.sibling_chunks)
        total += sum(chunk.token_count for chunk in self.import_chunks)
        total += sum(chunk.token_count for chunk in self.child_sections)

        return total

    @property
    def citation(self) -> str:
        """Generate citation for base chunk.

        Returns:
            Citation string in format: "file_path:start_line-end_line"
        """
        return f"{self.base_chunk.file_path}:{self.base_chunk.start_line}-{self.base_chunk.end_line}"

    def get_full_context(self) -> str:
        """Assemble full context string for synthesis.

        Combines base chunk with expanded context in a structured format.

        Returns:
            Multi-line string with all context formatted for LLM consumption
        """
        parts = []

        # Parent context
        if self.parent_chunk:
            parts.append(f"# Parent Context ({self.parent_chunk.name})")
            parts.append(self.parent_chunk.content)
            parts.append("")

        # Base chunk
        parts.append(f"# Main Chunk ({self.base_chunk.name})")
        parts.append(self.base_chunk.content)
        parts.append("")

        # Siblings
        if self.sibling_chunks:
            parts.append(f"# Related Methods ({len(self.sibling_chunks)} siblings)")
            for sibling in self.sibling_chunks:
                parts.append(f"## {sibling.name}")
                parts.append(sibling.content)
                parts.append("")

        # Imports
        if self.import_chunks:
            parts.append(f"# Dependencies ({len(self.import_chunks)} imports)")
            for imp in self.import_chunks:
                parts.append(f"## {imp.name}")
                parts.append(imp.content)
                parts.append("")

        # Child sections
        if self.child_sections:
            parts.append(f"# Sub-sections ({len(self.child_sections)} children)")
            for child in self.child_sections:
                parts.append(f"## {child.name}")
                parts.append(child.content)
                parts.append("")

        return "\n".join(parts)
