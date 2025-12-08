"""Conversation memory node (simplified with LangGraph built-ins).

This node handles end-of-turn operations. Conversation history is now
managed automatically by LangGraph's `add_messages` annotation and
the checkpointer (MemorySaver).

Author: Hay Hoffman
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.state import AgentState

__all__ = ["conversation_memory_node"]

logger = logging.getLogger(__name__)


def conversation_memory_node(state: AgentState) -> dict[str, Any]:
    """Handle end-of-turn operations (simplified v1.2).

    v1.2: Conversation history is now managed by LangGraph's `add_messages`
    annotation. This node only needs to:
    1. Add the current turn's messages to the state (auto-accumulated)
    2. Reset per-query state for the next turn

    The checkpointer (MemorySaver) automatically persists messages across
    invocations when using the same thread_id.

    Args:
        state: Current agent state containing:
            - user_query: User's question
            - final_answer: Generated answer
            - messages: Existing conversation (managed by add_messages)

    Returns:
        Updated state with:
            - messages: Current turn added (HumanMessage + AIMessage)
            - (cleared): per-query state for next turn

    Example:
        >>> state = {"user_query": "How does BM25 work?", "final_answer": "BM25 uses..."}
        >>> result = conversation_memory_node(state)
        >>> len(result["messages"])  # New messages for this turn
        2
    """
    user_query: str = state.get("user_query") or ""
    final_answer: str = state.get("final_answer") or ""

    # Build messages for this turn
    # add_messages annotation will automatically append these to existing messages
    turn_messages = [
        HumanMessage(content=user_query),
        AIMessage(content=final_answer),
    ]

    # Count turns from messages (each turn = 1 human + 1 AI message)
    existing_messages = state.get("messages") or []
    turn_count = (len(existing_messages) // 2) + 1

    logger.info(
        f"[SUCCESS] Completed turn #{turn_count}: "
        f"query={user_query[:50]}..., answer_length={len(final_answer)}"
    )

    # Return updates - messages will accumulate via add_messages
    # Clear per-query state for next invocation
    return {
        "messages": turn_messages,  # add_messages will append, not replace
        # Clear per-query state
        "router_output": None,
        "retrieved_chunks": None,
        "expanded_chunks": None,
        "retrieval_metadata": None,
        "final_answer": None,
        "citations": None,
        "error": None,
    }
