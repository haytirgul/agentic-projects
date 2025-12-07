"""Intent classification models for query preprocessing.

This module defines the structured output model for LLM-powered intent
classification, used when regex patterns don't match and conversation
history exists.

Author: Hay Hoffman
Version: 1.0
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["QueryIntent", "IntentType"]


class IntentType(str, Enum):
    """Types of user query intents."""

    HISTORY_REQUEST = "history_request"
    """User wants to see conversation history (e.g., 'what did I ask earlier?')"""

    FOLLOW_UP = "follow_up"
    """Query references previous context and history is sufficient to answer
    (e.g., 'tell me more about that')"""

    FOLLOW_UP_WITH_RETRIEVAL = "follow_up_with_retrieval"
    """Query references previous context BUT adds new requirements that need
    fresh retrieval (e.g., 'also show me the error handling for that')"""

    NEW_QUESTION = "new_question"
    """Independent question about the HTTPX codebase that needs retrieval"""

    GENERAL_QUESTION = "general_question"
    """General question NOT about httpx codebase - can be answered from LLM knowledge
    (e.g., 'hello', 'what is a class?', 'explain async/await')"""

    OUT_OF_SCOPE = "out_of_scope"
    """Query contains code that does NOT use HTTPX library - outside agent's scope
    (e.g., code using requests, aiohttp, urllib, or custom clients like 'GitHubClient')"""


class QueryIntent(BaseModel):
    """Structured output for query intent classification.

    Used by the LLM to classify user queries and resolve references
    to previous conversation context.

    Attributes:
        intent: The classified intent type
        references_previous: Whether query contains pronouns/references to history
        needs_retrieval: Whether fresh retrieval is required to answer
        resolved_query: Query with resolved references (if follow_up)
        reasoning: Brief explanation of classification decision

    Example:
        >>> # Follow-up that can be answered from history
        >>> intent = QueryIntent(
        ...     intent="follow_up",
        ...     references_previous=True,
        ...     needs_retrieval=False,
        ...     resolved_query="How does BM25 tokenization split identifiers?",
        ...     reasoning="User said 'that' referring to BM25 from previous turn"
        ... )
        >>> # Follow-up that needs new information
        >>> intent = QueryIntent(
        ...     intent="follow_up_with_retrieval",
        ...     references_previous=True,
        ...     needs_retrieval=True,
        ...     resolved_query="Show me the error handling in HTTPClient",
        ...     reasoning="User asks about HTTPClient (from history) but wants error handling (new info)"
        ... )
    """

    intent: Literal[
        "history_request", "follow_up", "follow_up_with_retrieval", "new_question", "general_question", "out_of_scope"
    ] = Field(description="The classified intent type")

    references_previous: bool = Field(
        description="Whether the query contains references to previous conversation"
    )

    needs_retrieval: bool = Field(
        description=(
            "Whether fresh retrieval from the codebase is required. "
            "True for: new_question, follow_up_with_retrieval. "
            "False for: history_request, follow_up, general_question."
        )
    )

    resolved_query: str = Field(
        description=(
            "The query with pronouns/references resolved. "
            "If intent is 'history_request', this is the original query. "
            "If intent is 'new_question', this is the cleaned query. "
            "If intent is 'follow_up' or 'follow_up_with_retrieval', "
            "this includes context from history."
        )
    )

    reasoning: str = Field(
        description="Brief explanation of why this intent was classified"
    )
