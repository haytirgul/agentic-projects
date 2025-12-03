"""Pydantic models for RAG document indexing."""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """A section within a document."""

    title: str = Field(description="Section heading text")
    level: int = Field(description="Heading level (1-6, where 1 is H1)")
    content: str = Field(description="Concatenated content from all content blocks")
    content_blocks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Original content blocks for reference"
    )


class Document(BaseModel):
    """An in-memory document from the knowledge base."""

    file_path: str = Field(description="Relative path to the JSON file")
    title: str = Field(description="Document title")
    framework: str = Field(description="Framework (langchain, langgraph, langsmith)")
    language: str = Field(description="Language (python, javascript, or empty)")
    topic: str = Field(description="Topic directory (concepts, how-tos, etc.)")
    sections: List[DocumentSection] = Field(
        default_factory=list,
        description="Flattened list of all sections in the document"
    )
    raw_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Original JSON data for reference"
    )


class RetrievedDocument(BaseModel):
    """A document retrieved from the knowledge base with granular content."""

    file_path: str = Field(description="Relative path to the JSON file")
    title: str = Field(description="Document title")
    framework: str = Field(description="Framework (langchain, langgraph, langsmith)")
    language: str = Field(description="Language (python, javascript, or empty)")
    topic: str = Field(description="Topic directory (concepts, how-tos, etc.)")

    # Granular retrieval fields
    granularity: Literal["document", "section", "content_block"] = Field(
        default="document",
        description="Level of granularity for this retrieval"
    )
    matched_text: str = Field(
        default="",
        description="The specific text that matched the query"
    )

    # Optional fields based on granularity
    section_title: Optional[str] = Field(
        default=None,
        description="Section title (for section/content_block granularity)"
    )
    section_level: Optional[int] = Field(
        default=None,
        description="Section level (for section/content_block granularity)"
    )
    content_block: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Specific content block that matched (for content_block granularity)"
    )

    # Legacy field for backward compatibility
    sections: List[DocumentSection] = Field(
        default_factory=list,
        description="Full sections (only populated for document-level retrieval)"
    )

    relevance_score: float = Field(
        ge=0, le=100,
        description="Relevance score from hybrid retrieval (0-100)"
    )
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the retrieval process"
    )


class WebSearchQueries(BaseModel):
    """Structured output for web search query generation."""

    queries: List[str] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="List of 1-3 optimized search queries"
    )
    reasoning: str = Field(
        ...,
        max_length=200,
        description="Brief explanation of query generation strategy"
    )

    @property
    def query_count(self) -> int:
        """Get the number of queries generated."""
        return len(self.queries)
