"""State reset node for clearing per-query state while preserving memory.

This node prepares the state for the next conversation turn.
"""

from typing import Any, Dict
import logging

from src.graph.state import AgentState

__all__ = ["reset_for_next_turn_node"]

logger = logging.getLogger(__name__)


def reset_for_next_turn_node(state: AgentState) -> Dict[str, Any]:
    """
    Reset per-query state while preserving conversation memory.

    This node:
    1. Clears per-query fields (user_input, cleaned_request, messages, etc.)
    2. Preserves conversation_memory
    3. Prepares state for next user input

    What gets cleared:
    - user_input, cleaned_request, code_snippets, extracted_data
    - intent_result (context is preserved in conversation memory)
    - retrieved_documents, retrieval_metadata
    - messages, final_response

    What gets preserved:
    - conversation_memory
    - continue_conversation (will be checked by routing)

    Input state:
        - conversation_memory: ConversationMemory
        - (all other fields to be cleared)

    Output state:
        - user_input: "" (cleared)
        - cleaned_request: None (cleared)
        - code_snippets: None (cleared)
        - extracted_data: None (cleared)
        - intent_result: None (cleared)
        - retrieved_documents: None (cleared)
        - retrieval_metadata: None (cleared)
        - messages: [] (cleared)
        - final_response: None (cleared)
        - conversation_memory: ConversationMemory (preserved)

    Example:
        >>> state = {
        ...     "user_input": "Previous query",
        ...     "final_response": "Previous response",
        ...     "conversation_memory": ConversationMemory(turns=[...])
        ... }
        >>> result = reset_for_next_turn_node(state)
        >>> result["user_input"]
        ""
        >>> result["conversation_memory"]  # Preserved
        ConversationMemory(turns=[...])
    """
    conversation_memory = state.get("conversation_memory")

    logger.info("Resetting state for next conversation turn...")
    logger.info(
        f"Preserved conversation memory with {len(conversation_memory.turns) if conversation_memory else 0} turns"
    )

    # Return cleared state with preserved memory
    return {
        # Clear per-query fields
        "user_input": "",
        "cleaned_request": None,
        "code_snippets": None,
        "extracted_data": None,
        "intent_result": None,
        "retrieved_documents": None,
        "retrieval_metadata": None,
        "messages": [],
        "final_response": None,
        # Preserve conversation memory
        "conversation_memory": conversation_memory,
        # Continue flag will be evaluated by routing
    }
