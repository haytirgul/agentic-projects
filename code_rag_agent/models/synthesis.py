"""Synthesis models for answer generation and citation extraction.

This module defines the data structures used by the Synthesizer node to generate
grounded answers with precise citations from retrieved code chunks.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path

__all__ = ["SynthesisRequest", "Citation", "SynthesizedAnswer", "SynthesisMetadata"]


class SynthesisRequest(BaseModel):
    """Input to the Synthesizer node containing retrieval results and query context.

    Combines the original query, retrieval results, and synthesis parameters.
    """

    user_query: str = Field(
        ...,
        description="Original user question"
    )
    retrieval_result: 'RetrievalResult' = Field(
        ...,
        description="Complete retrieval results with chunks and metadata"
    )
    max_answer_length: int = Field(
        2000,
        ge=100,
        le=8000,
        description="Maximum answer length in characters"
    )
    include_raw_chunks: bool = Field(
        False,
        description="Whether to include raw chunk data in output"
    )
    synthesis_style: str = Field(
        "concise",
        description="Answer style: concise, detailed, explanatory",
        pattern="^(concise|detailed|explanatory)$"
    )

    @property
    def available_chunks(self) -> List['RetrievedChunk']:
        """Get all retrieved chunks."""
        return self.retrieval_result.chunks

    @property
    def top_chunks(self, n: int = 5) -> List['RetrievedChunk']:
        """Get top N chunks by relevance score."""
        return sorted(self.available_chunks, key=lambda x: x.final_score, reverse=True)[:n]

    @property
    def query_type(self) -> str:
        """Extract query type from retrieval metadata if available."""
        # This would be passed from the router decision
        return getattr(self.retrieval_result.request.router_decision.analysis, 'query_type', 'general')

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class Citation(BaseModel):
    """A citation referencing specific code in the repository.

    Represents a precise reference to code that supports an answer claim.
    """

    file_path: Path = Field(
        ...,
        description="Path to the file containing the cited code"
    )
    start_line: int = Field(
        ...,
        ge=1,
        description="Starting line number of the citation"
    )
    end_line: int = Field(
        ...,
        ge=1,
        description="Ending line number of the citation"
    )
    chunk_type: str = Field(
        ...,
        description="Type of code chunk: function, class, method, etc."
    )
    parent_context: str = Field(
        "",
        description="Parent class or module context"
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How relevant this citation is to the claim"
    )
    excerpt: str = Field(
        ...,
        description="Brief excerpt of the cited code (1-2 lines)"
    )
    claim_supported: str = Field(
        ...,
        description="Brief description of what claim this citation supports"
    )

    @property
    def citation_string(self) -> str:
        """Generate standard citation format: path/to/file.py:line_start-line_end"""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    @property
    def display_citation(self) -> str:
        """Generate user-friendly citation with context."""
        context = f" ({self.parent_context})" if self.parent_context else ""
        return f"{self.file_path}:{self.start_line}-{self.end_line}{context}"

    def matches_chunk(self, chunk: 'RetrievedChunk') -> bool:
        """Check if this citation matches a retrieved chunk."""
        return (
            self.file_path == chunk.file_path and
            self.start_line == chunk.start_line and
            self.end_line == chunk.end_line
        )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            Path: str
        }


class SynthesisMetadata(BaseModel):
    """Metadata about the synthesis process.

    Tracks how the answer was generated and validated.
    """

    synthesis_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the synthesis was completed"
    )
    chunks_used: int = Field(
        0,
        description="Number of chunks referenced in the answer"
    )
    total_chunks_available: int = Field(
        0,
        description="Total chunks available from retrieval"
    )
    citations_generated: int = Field(
        0,
        description="Number of citations created"
    )
    answer_length: int = Field(
        0,
        description="Length of generated answer in characters"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the synthesized answer"
    )
    grounding_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the answer is grounded in retrieved chunks"
    )

    # Quality metrics
    hallucination_risk: str = Field(
        "low",
        description="Risk assessment: low, medium, high",
        pattern="^(low|medium|high)$"
    )
    citation_completeness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of claims that have citations"
    )
    answer_relevance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the answer addresses the original query"
    )

    # Processing details
    synthesis_strategy: str = Field(
        "standard",
        description="Synthesis approach used"
    )
    fallback_applied: bool = Field(
        False,
        description="Whether fallback strategies were used"
    )
    processing_warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during synthesis"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SynthesizedAnswer(BaseModel):
    """Complete synthesized answer with citations and metadata.

    Represents the final output of the synthesis process - a natural language
    answer grounded in retrieved code with precise citations.
    """

    request: SynthesisRequest = Field(
        ...,
        description="Original synthesis request"
    )
    answer: str = Field(
        ...,
        description="Generated answer text"
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer"
    )
    metadata: SynthesisMetadata = Field(
        ...,
        description="Metadata about the synthesis process"
    )

    @property
    def is_grounded(self) -> bool:
        """Check if the answer is properly grounded in citations."""
        return len(self.citations) > 0 and self.metadata.grounding_score > 0.7

    @property
    def citation_summary(self) -> str:
        """Generate a summary of all citations."""
        if not self.citations:
            return "No citations available"

        paths = {}
        for citation in self.citations:
            path_str = str(citation.file_path)
            if path_str not in paths:
                paths[path_str] = []
            paths[path_str].append(f"{citation.start_line}-{citation.end_line}")

        summary_parts = []
        for path, ranges in paths.items():
            summary_parts.append(f"{path}:{', '.join(ranges)}")

        return "; ".join(summary_parts)

    @property
    def answer_with_citations(self) -> str:
        """Generate the answer with inline citations."""
        answer_text = self.answer

        # Add citation references at the end
        if self.citations:
            citation_lines = ["\n\n**Citations:**"]
            for i, citation in enumerate(self.citations, 1):
                citation_lines.append(f"{i}. {citation.display_citation}")
            answer_text += "\n".join(citation_lines)

        return answer_text

    def get_citations_for_claim(self, claim_text: str) -> List[Citation]:
        """Find citations that support a specific claim."""
        return [
            citation for citation in self.citations
            if claim_text.lower() in citation.claim_supported.lower()
        ]

    def validate_citations(self) -> Dict[str, Any]:
        """Validate that all citations reference actual retrieved chunks."""
        validation_results = {
            "total_citations": len(self.citations),
            "valid_citations": 0,
            "invalid_citations": 0,
            "missing_chunks": [],
            "validation_warnings": []
        }

        available_chunks = self.request.available_chunks

        for citation in self.citations:
            chunk_matches = [
                chunk for chunk in available_chunks
                if citation.matches_chunk(chunk)
            ]

            if chunk_matches:
                validation_results["valid_citations"] += 1
            else:
                validation_results["invalid_citations"] += 1
                validation_results["missing_chunks"].append(citation.citation_string)
                validation_results["validation_warnings"].append(
                    f"Citation {citation.citation_string} does not match any retrieved chunk"
                )

        return validation_results

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
