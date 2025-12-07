"""State schema for the Code RAG Agent.

This module defines the shared state structure used across all graph nodes
in the retrieval and synthesis pipeline.

Author: Hay Hoffman
Version: 1.2
"""

from typing import TypedDict

from langchain_core.messages import BaseMessage
from models.retrieval import RouterOutput

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    """State schema for the Code RAG Agent.

    This state is shared across all nodes in the LangGraph workflow.
    Each node can read from and write to this state.

    Attributes:
        # User input
        user_query: The user's original code question

        # Security validation
        gateway_passed: Whether request passed security validation
        gateway_reason: Reason for security validation failure
        is_blocked: Whether request was blocked by security

        # Input preprocessing (v1.2, v1.3, v1.4)
        cleaned_query: Query after filler removal and reference resolution
        is_history_query: True if user asks about conversation history
        is_follow_up: True if query references previous context
        is_general_question: True if query is general (no retrieval needed) (v1.3)
        is_out_of_scope: True if query contains non-HTTPX code (v1.4)
        needs_retrieval: True if fresh retrieval is needed (v1.3)

        # Router output
        router_output: Query decomposition and retrieval planning (RouterOutput)
        codebase_tree: Filtered directory tree for router context

        # Retrieval results
        retrieved_chunks: list of retrieved chunks with RRF scores (RetrievedChunk)
        expanded_chunks: Chunks with context expansion (ExpandedChunk)
        retrieval_metadata: Debugging metadata (chunk counts, scores, etc.)

        # Synthesis
        messages: Conversation messages for LLM synthesis
        final_answer: The synthesized answer with citations
        citations: list of code citations (file:line references)

        # Conversation history
        conversation_history: list of previous (query, answer) pairs
        turn_count: Number of turns in current conversation

        # Error handling
        error: Error message if pipeline fails

    Example:
        >>> from models.retrieval import RouterOutput, RetrievalRequest
        >>> state = AgentState(
        ...     user_query="How does BM25 tokenization work?",
        ...     gateway_passed=True,
        ...     router_output=RouterOutput(
        ...         cleaned_query="BM25 tokenization implementation",
        ...         retrieval_requests=[
        ...             RetrievalRequest(
        ...                 query="BM25 tokenize split camelCase",
        ...                 source_types=["code"],
        ...                 folders=["src/retrieval/"],
        ...                 file_patterns=["hybrid_retriever.py"],
        ...                 reasoning="Need BM25 tokenization logic"
        ...             )
        ...         ]
        ...     )
        ... )
    """

    # User input
    user_query: str

    # Security validation
    gateway_passed: bool | None
    gateway_reason: str | None
    is_blocked: bool | None

    # Input preprocessing (v1.2, v1.3, v1.4)
    cleaned_query: str | None
    is_history_query: bool | None
    is_follow_up: bool | None
    is_general_question: bool | None  # v1.3: General question (no retrieval needed)
    is_out_of_scope: bool | None  # v1.4: Query about non-HTTPX code
    needs_retrieval: bool | None  # v1.3: Whether fresh retrieval is needed

    # Router output (query decomposition)
    router_output: RouterOutput | None

    # Retrieval results
    retrieved_chunks: list[dict] | None  # Serialized RetrievedChunk objects
    expanded_chunks: list[dict] | None  # Serialized ExpandedChunk objects
    retrieval_metadata: dict | None

    # Synthesis
    messages: list[BaseMessage]
    final_answer: str | None
    citations: list[dict] | None

    # Conversation history (for follow-up questions)
    conversation_history: list[dict] | None  # list of {query, answer, timestamp}
    turn_count: int | None

    # Error handling
    error: str | None
