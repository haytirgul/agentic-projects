"""Retrieval models for code search and result processing.

This module defines the data structures used by the Retriever node to execute
hybrid search and return relevant code chunks with ranking.
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from pathlib import Path

__all__ = ["RetrievalRequest", "RetrievedChunk", "RetrievalResult", "RetrievalMetadata"]


class RetrievalRequest(BaseModel):
    """Input to the Retriever node containing search parameters.

    Combines router analysis with execution parameters for the retrieval process.
    """

    user_query: str = Field(
        ...,
        description="Original user query"
    )
    router_decision: 'RouterDecision' = Field(
        ...,
        description="Complete router decision with analysis and strategy"
    )
    top_k: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of top chunks to retrieve"
    )
    include_metadata: bool = Field(
        True,
        description="Whether to include detailed metadata in results"
    )
    rerank: bool = Field(
        True,
        description="Whether to apply reranking to results"
    )

    @property
    def primary_terms(self) -> List[str]:
        """Get primary search terms from router strategy."""
        return self.router_decision.strategy.primary_search_terms

    @property
    def secondary_terms(self) -> List[str]:
        """Get secondary search terms from router strategy."""
        return self.router_decision.strategy.secondary_search_terms

    @property
    def module_filters(self) -> List[str]:
        """Get module filters from router strategy."""
        return self.router_decision.strategy.module_filters

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class RetrievedChunk(BaseModel):
    """A code chunk retrieved from the vector search.

    Represents a single code chunk with its content, location, and relevance scores.
    """

    content: str = Field(
        ...,
        description="The actual code content of this chunk"
    )
    file_path: Path = Field(
        ...,
        description="Relative path from repository root"
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="Starting line number in the file"
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="Ending line number in the file"
    )
    chunk_type: str = Field(
        ...,
        description="Type of code chunk: function, class, method, etc."
    )
    parent_context: str = Field(
        "",
        description="Parent class or module context"
    )
    docstring: Optional[str] = Field(
        None,
        description="Extracted docstring if available"
    )

    # Relevance scores
    dense_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Dense embedding similarity score"
    )
    sparse_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Sparse keyword matching score"
    )
    reranked_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Score after reranking"
    )
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final relevance score used for ranking"
    )

    # Metadata
    module_name: str = Field(
        ...,
        description="Module name (e.g., _client, _ssl)"
    )
    language: str = Field(
        "python",
        description="Programming language"
    )
    search_terms_matched: List[str] = Field(
        default_factory=list,
        description="Which search terms were matched in this chunk"
    )
    retrieval_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this chunk was retrieved"
    )

    @property
    def relevance_score(self) -> float:
        """Get the most relevant score for this chunk."""
        return self.final_score

    @property
    def citation(self) -> str:
        """Generate citation string for this chunk."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def preview_text(self) -> str:
        """Get a preview of the chunk content (first 200 chars)."""
        if len(self.content) <= 200:
            return self.content
        return self.content[:200] + "..."

    def matches_module_filter(self, allowed_modules: List[str]) -> bool:
        """Check if this chunk matches any of the allowed modules."""
        if not allowed_modules:
            return True
        return any(module in self.module_name for module in allowed_modules)

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }


class RetrievalMetadata(BaseModel):
    """Metadata about the retrieval process.

    Tracks statistics and performance of the retrieval operation.
    """

    total_chunks_searched: int = Field(
        0,
        description="Total number of chunks in the search index"
    )
    chunks_retrieved: int = Field(
        0,
        description="Number of chunks returned"
    )
    search_time_seconds: float = Field(
        0.0,
        ge=0.0,
        description="Time taken for the search operation"
    )
    dense_search_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time for dense vector search"
    )
    sparse_search_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time for sparse keyword search"
    )
    reranking_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Time for reranking operation"
    )

    search_strategy_used: str = Field(
        "hybrid",
        description="Search strategy: dense, sparse, or hybrid"
    )
    reranking_applied: bool = Field(
        False,
        description="Whether reranking was applied"
    )
    filters_applied: List[str] = Field(
        default_factory=list,
        description="List of filters that were applied"
    )
    top_k_requested: int = Field(
        ...,
        description="Original top_k value requested"
    )

    # Performance metrics
    average_dense_score: Optional[float] = Field(
        None,
        description="Average dense similarity score"
    )
    average_sparse_score: Optional[float] = Field(
        None,
        description="Average sparse similarity score"
    )
    score_distribution: Optional[dict] = Field(
        None,
        description="Distribution of final scores"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class RetrievalResult(BaseModel):
    """Complete result of a retrieval operation.

    Contains the retrieved chunks, metadata, and performance information.
    """

    request: RetrievalRequest = Field(
        ...,
        description="Original retrieval request"
    )
    chunks: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Retrieved code chunks, sorted by relevance"
    )
    metadata: RetrievalMetadata = Field(
        ...,
        description="Metadata about the retrieval process"
    )
    retrieval_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the retrieval was completed"
    )

    @property
    def is_successful(self) -> bool:
        """Check if retrieval returned any results."""
        return len(self.chunks) > 0

    @property
    def top_chunk(self) -> Optional[RetrievedChunk]:
        """Get the highest-ranked chunk."""
        return self.chunks[0] if self.chunks else None

    @property
    def average_score(self) -> float:
        """Calculate average final score across all chunks."""
        if not self.chunks:
            return 0.0
        return sum(chunk.final_score for chunk in self.chunks) / len(self.chunks)

    @property
    def citations(self) -> List[str]:
        """Get all citations for the retrieved chunks."""
        return [chunk.citation for chunk in self.chunks]

    def chunks_by_module(self) -> dict:
        """Group chunks by module name."""
        result = {}
        for chunk in self.chunks:
            if chunk.module_name not in result:
                result[chunk.module_name] = []
            result[chunk.module_name].append(chunk)
        return result

    def get_top_n_chunks(self, n: int) -> List[RetrievedChunk]:
        """Get top N chunks by relevance score."""
        return sorted(self.chunks, key=lambda x: x.final_score, reverse=True)[:n]

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: str
        }
