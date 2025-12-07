"""Edge routing functions for conditional graph navigation.

This module contains routing logic that determines which node to execute next
based on the current state of the graph.

Architecture Version: 1.2
Author: Hay Hoffman
"""

import logging
from src.agent.state import AgentState

__all__ = [
    "route_after_security_gateway",
    "route_after_input_preprocessor",
    "route_after_conversation_memory",
]

logger = logging.getLogger(__name__)


def route_after_security_gateway(state: AgentState) -> str:
    """Route after security gateway: end for blocked or continue to preprocessor.

    Args:
        state: Current graph state with gateway_passed and is_blocked flags

    Returns:
        "END" if blocked, "input_preprocessor" otherwise
    """
    gateway_passed = state.get("gateway_passed", True)
    is_blocked = state.get("is_blocked", False)

    if not gateway_passed or is_blocked:
        gateway_reason = state.get("gateway_reason", "Security violation")
        logger.warning(f"Security gateway blocked: {gateway_reason}")
        return "END"

    logger.debug("Security passed, continuing to input_preprocessor")
    return "input_preprocessor"


def route_after_input_preprocessor(state: AgentState) -> str:
    """Route after input preprocessor based on retrieval needs.

    Routing decision:
    - history_request: Skip to memory (final_answer already set)
    - out_of_scope: Skip to memory (final_answer already set with rejection message)
    - general_question: Skip to synthesis (LLM can answer from knowledge)
    - follow_up (no retrieval): Skip to synthesis (use history context)
    - follow_up_with_retrieval / new_question: Go to router for fresh retrieval

    Args:
        state: Current graph state with is_history_query, is_out_of_scope, is_general_question, needs_retrieval flags

    Returns:
        "conversation_memory" if history query or out_of_scope (final_answer set),
        "synthesis" if general question or follow_up with sufficient history (needs_retrieval=False),
        "router" otherwise (needs fresh retrieval)

    Example:
        >>> state = {"is_history_query": True}
        >>> route_after_input_preprocessor(state)
        "conversation_memory"

        >>> state = {"is_out_of_scope": True}
        >>> route_after_input_preprocessor(state)
        "conversation_memory"

        >>> state = {"is_general_question": True, "needs_retrieval": False}
        >>> route_after_input_preprocessor(state)
        "synthesis"

        >>> state = {"is_follow_up": True, "needs_retrieval": False}
        >>> route_after_input_preprocessor(state)
        "synthesis"

        >>> state = {"needs_retrieval": True}
        >>> route_after_input_preprocessor(state)
        "router"
    """
    is_history_query = state.get("is_history_query", False)
    is_out_of_scope = state.get("is_out_of_scope", False)
    is_general_question = state.get("is_general_question", False)
    needs_retrieval = state.get("needs_retrieval", True)

    if is_history_query:
        logger.info("History query detected - skipping to conversation_memory")
        return "conversation_memory"

    if is_out_of_scope:
        logger.info("Out of scope query (non-HTTPX code) - skipping to conversation_memory")
        return "conversation_memory"

    if is_general_question:
        logger.info("General question detected - skipping to synthesis (no retrieval needed)")
        return "synthesis"

    if not needs_retrieval:
        # Follow-up that can be answered from history context
        logger.info("Follow-up detected (history sufficient) - skipping to synthesis")
        return "synthesis"

    logger.debug("Retrieval needed - continuing to router")
    return "router"


def route_after_conversation_memory(state: AgentState) -> str:
    """Route after conversation_memory: restart or end.

    Args:
        state: Current graph state with continue_conversation flag

    Returns:
        "security_gateway" if continuing (new query needs validation),
        "END" otherwise
    """
    continue_conv = state.get("continue_conversation", False)

    if continue_conv:
        logger.debug("Continuing conversation, routing to security_gateway")
        return "security_gateway"

    logger.debug("Ending conversation")
    return "END"
