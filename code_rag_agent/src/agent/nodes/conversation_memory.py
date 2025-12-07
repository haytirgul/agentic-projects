"""Conversation memory node (combined: save + prompt + reset).

This node handles all conversation memory operations in a single node
for minimal latency:
- Saves conversation turn (query, answer, citations)
- Prompts user to continue
- Resets per-query state while preserving history

Author: Hay Hoffman
Version: 1.1
"""

import logging
from datetime import datetime
from typing import Any

from src.agent.state import AgentState

__all__ = ["conversation_memory_node"]

logger = logging.getLogger(__name__)


def conversation_memory_node(state: AgentState) -> dict[str, Any]:
    """Handle conversation memory (save + prompt + reset combined).

    This node performs three operations in one for minimal latency:
    1. Save turn: Store (query, answer, citations) in conversation_history
    2. Prompt continue: Check if user wants to continue (external control)
    3. Reset state: Clear per-query state, preserve history

    Args:
        state: Current agent state containing:
            - user_query: User's question
            - final_answer: Generated answer
            - citations: list of citations
            - conversation_history: Existing history (optional)
            - continue_conversation: External control flag (optional)

    Returns:
        Updated state with:
            - conversation_history: Updated history with new turn
            - turn_count: Incremented turn count
            - continue_conversation: Whether to continue or end
            - (cleared): router_output, retrieved_chunks, expanded_chunks, etc.

    Example:
        >>> state = {
        ...     "user_query": "How does BM25 work?",
        ...     "final_answer": "BM25 uses...",
        ...     "citations": [...]
        ... }
        >>> result = conversation_memory_node(state)
        >>> len(result["conversation_history"])
        1
    """
    # Extract turn data
    user_query: str = state.get("user_query") or ""
    final_answer: str = state.get("final_answer") or ""
    citations: list = state.get("citations") or []

    # Get existing history
    conversation_history: list[dict] = state.get("conversation_history") or []
    turn_count: int = state.get("turn_count") or 0

    # === OPERATION 1: Save Turn ===
    turn_data = {
        "query": user_query,
        "answer": final_answer,
        "citations": citations,
        "timestamp": datetime.now().isoformat(),
        "turn": turn_count + 1,
    }

    conversation_history.append(turn_data)
    turn_count += 1

    logger.info(
        f"[SUCCESS] Saved conversation turn #{turn_count}: "
        f"query={user_query[:50]}..., answer_length={len(final_answer)}"
    )

    # === OPERATION 2: Prompt Continue ===
    # Check for external control (chatbot mode)
    external_control = state.get("continue_conversation") is False

    if external_control:
        logger.info("External control detected (chatbot mode) - ending turn")
        continue_conversation = False
    else:
        # Autonomous mode: continue by default
        logger.info("Autonomous mode - ready for next query")
        continue_conversation = True

    # === OPERATION 3: Reset State ===
    # Clear per-query state while preserving history
    reset_updates = {
        # Preserve
        "conversation_history": conversation_history,
        "turn_count": turn_count,
        "continue_conversation": continue_conversation,
        # Clear per-query state
        "router_output": None,
        "codebase_tree": None,
        "retrieved_chunks": None,
        "expanded_chunks": None,
        "retrieval_metadata": None,
        "messages": [],
        "final_answer": None,
        "citations": None,
        "error": None,
    }

    logger.debug(f"State reset complete (preserved history: {turn_count} turns)")

    return reset_updates
