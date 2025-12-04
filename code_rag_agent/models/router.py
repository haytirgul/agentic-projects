"""Router models for query analysis and retrieval planning.

This module defines the data structures used by the Router node to analyze
user queries and plan retrieval strategies for the httpx codebase.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

__all__ = ["QueryAnalysis", "RetrievalStrategy", "RouterDecision"]


class QueryAnalysis(BaseModel):
    """Analysis of user query components and intent.

    Breaks down the query into actionable components for retrieval planning.
    """

    query_type: str = Field(
        ...,
        description="Type of query: 'behavior', 'location', 'comparison', 'explanation', 'general'",
        pattern="^(behavior|location|comparison|explanation|general)$"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Key terms and concepts extracted from query",
        min_items=1
    )
    modules_of_interest: List[str] = Field(
        default_factory=list,
        description="Specific httpx modules mentioned or implied"
    )
    concepts: List[str] = Field(
        default_factory=list,
        description="High-level concepts (SSL, proxy, timeout, etc.)"
    )
    is_followup: bool = Field(
        False,
        description="Whether this appears to be a follow-up question"
    )
    context_references: List[str] = Field(
        default_factory=list,
        description="References to previous conversation context"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis (0.0-1.0)"
    )


class RetrievalStrategy(BaseModel):
    """Strategy for retrieving relevant code chunks.

    Defines how to search for and filter code chunks based on query analysis.
    """

    primary_search_terms: List[str] = Field(
        default_factory=list,
        description="Main terms to search for in code",
        min_items=1
    )
    secondary_search_terms: List[str] = Field(
        default_factory=list,
        description="Additional terms for broader search"
    )
    module_filters: List[str] = Field(
        default_factory=list,
        description="Specific modules to focus search on"
    )
    code_patterns: List[str] = Field(
        default_factory=list,
        description="Specific code patterns to look for (class names, function names, etc.)"
    )
    search_scope: str = Field(
        "broad",
        description="Search scope: 'narrow', 'medium', 'broad'",
        pattern="^(narrow|medium|broad)$"
    )
    expected_chunk_types: List[str] = Field(
        default_factory=lambda: ["function", "class", "method"],
        description="Types of code chunks most relevant to this query"
    )
    priority_modules: List[str] = Field(
        default_factory=list,
        description="Modules to prioritize in search results"
    )


class RouterDecision(BaseModel):
    """Complete router decision with analysis and strategy.

    Combines query analysis with retrieval planning for the agent to execute.
    """

    user_query: str = Field(
        ...,
        description="Original user query"
    )
    analysis: QueryAnalysis = Field(
        ...,
        description="Detailed analysis of the query"
    )
    strategy: RetrievalStrategy = Field(
        ...,
        description="Planned retrieval strategy"
    )
    estimated_complexity: str = Field(
        "medium",
        description="Estimated query complexity: 'simple', 'medium', 'complex'",
        pattern="^(simple|medium|complex)$"
    )
    reasoning: str = Field(
        ...,
        description="Explanation of how this query was analyzed and planned"
    )
    potential_challenges: List[str] = Field(
        default_factory=list,
        description="Potential challenges or edge cases to consider"
    )

    @property
    def is_behavior_query(self) -> bool:
        """Check if this is a behavior-focused query."""
        return self.analysis.query_type == "behavior"

    @property
    def is_location_query(self) -> bool:
        """Check if this is a location-focused query."""
        return self.analysis.query_type == "location"

    @property
    def needs_multiple_searches(self) -> bool:
        """Check if this query likely needs multiple retrieval attempts."""
        return (
            len(self.strategy.primary_search_terms) > 3 or
            self.estimated_complexity == "complex" or
            len(self.analysis.concepts) > 2
        )

    def get_search_terms_summary(self) -> str:
        """Get a human-readable summary of search terms."""
        primary = ", ".join(self.strategy.primary_search_terms[:3])
        if len(self.strategy.primary_search_terms) > 3:
            primary += f" (+{len(self.strategy.primary_search_terms) - 3} more)"

        if self.strategy.secondary_search_terms:
            secondary = ", ".join(self.strategy.secondary_search_terms[:2])
            return f"{primary} (also: {secondary})"

        return primary
