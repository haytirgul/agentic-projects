"""Edge routing functions for conditional graph navigation.

This module contains routing logic that determines which node to execute next
based on the current state of the graph.
"""

import logging
from .state import AgentState
from src.llm.workers import has_pending_tool_calls

__all__ = [
    "route_after_security_gateway",
    "route_after_preprocessing",
    "route_after_intent_classification",
    "route_after_agent",
    "route_after_prompt_continue",
]

logger = logging.getLogger(__name__)


def route_after_security_gateway(state: AgentState) -> str:
    """
    Route after security gateway: end conversation for blocked requests or continue to intent classification.

    This routing function checks if the security gateway blocked the request due to malicious content.
    If blocked, immediately end the conversation. Otherwise, continue to intent classification
    (which now includes preprocessing internally).

    Args:
        state: Current graph state with gateway_passed flag

    Returns:
        "END" if security gateway blocked the request,
        "intent_classification" otherwise

    Example:
        >>> state = {"gateway_passed": False, "gateway_reason": "Malicious content detected"}
        >>> route_after_security_gateway(state)
        "END"

        >>> state = {"gateway_passed": True}
        >>> route_after_security_gateway(state)
        "intent_classification"
    """
    gateway_passed = state.get("gateway_passed", True)

    if not gateway_passed:
        gateway_reason = state.get("gateway_reason", "Security violation")
        logger.warning(f"Security gateway blocked request, ending conversation: {gateway_reason}")
        return "END"
    else:
        logger.debug("Security gateway passed, continuing to intent_classification")
        return "intent_classification"


def route_after_preprocessing(state: AgentState) -> str:
    """
    Route after preprocessing: end conversation for malicious requests or continue to intent classification.

    This routing function checks if the preprocessing node detected suspicious patterns
    that indicate malicious intent. If suspicious patterns are found with high risk,
    immediately end the conversation. Otherwise, continue to intent classification.

    Args:
        state: Current graph state (may contain security analysis from preprocessing)

    Returns:
        "END" if preprocessing detected high-risk malicious patterns,
        "intent_classification" otherwise

    Example:
        >>> state = {"security_analysis": {"risk_level": "high", "suspicious_patterns": ["data_exfiltration"]}}
        >>> route_after_preprocessing(state)
        "END"

        >>> state = {"security_analysis": {"risk_level": "low", "suspicious_patterns": []}}
        >>> route_after_preprocessing(state)
        "intent_classification"
    """
    # Check if preprocessing detected suspicious patterns
    # The security analysis might be in different state keys depending on implementation
    security_analysis = None

    # Check various possible locations for security analysis
    if "security_analysis" in state:
        security_analysis = state["security_analysis"]
    elif "extracted_data" in state and isinstance(state["extracted_data"], dict) and "security_analysis" in state["extracted_data"]:
        security_analysis = state["extracted_data"]["security_analysis"]

    if security_analysis:
        risk_level = security_analysis.get("risk_level", "low")
        suspicious_patterns = security_analysis.get("suspicious_patterns", [])

        if risk_level == "high" or (risk_level == "medium" and len(suspicious_patterns) >= 2):
            logger.warning(
                f"Preprocessing detected malicious patterns, ending conversation: "
                f"risk_level={risk_level}, patterns={suspicious_patterns}"
            )
            return "END"

    logger.debug("Preprocessing completed without high-risk patterns, continuing to intent_classification")
    return "intent_classification"


def route_after_intent_classification(state: AgentState) -> str:
    """
    Route after intent classification: skip RAG for clarifications or proceed to RAG.

    The intent classification node handles tools internally, so it always returns
    a complete IntentClassification result. This function determines if RAG retrieval
    is needed based on that classification.

    Args:
        state: Current graph state with intent_result

    Returns:
        "prepare_messages" if RAG should be skipped (clarification),
        "hybrid_retrieval" otherwise

    Example:
        >>> state = {"intent_result": IntentClassification(conversation_context="clarification", requires_rag=False)}
        >>> route_after_intent_classification(state)
        "prepare_messages"

        >>> state = {"intent_result": IntentClassification(conversation_context="new_topic", requires_rag=True)}
        >>> route_after_intent_classification(state)
        "hybrid_retrieval"
    """
    intent_result = state.get("intent_result")

    if not intent_result:
        logger.warning("No intent result found, defaulting to RAG retrieval")
        return "hybrid_retrieval"

    # Check if RAG is required
    requires_rag = getattr(intent_result, "requires_rag", True)
    conversation_context = getattr(intent_result, "conversation_context", "new_topic")

    if not requires_rag and conversation_context == "clarification":
        logger.info(
            f"Skipping RAG retrieval for clarification (conversation_context={conversation_context})"
        )
        return "prepare_messages"
    else:
        logger.info(
            f"Proceeding to RAG retrieval (requires_rag={requires_rag}, "
            f"conversation_context={conversation_context})"
        )
        return "hybrid_retrieval"


def route_after_agent(state: AgentState) -> str:
    """
    Route after agent node: continue with tools or finalize.

    This routing function determines if the response agent needs to call tools
    (e.g., ask_user for clarification) or if it's ready to finalize the response.

    Args:
        state: Current graph state with messages

    Returns:
        "tools" if there are pending tool calls,
        "finalize" otherwise

    Example:
        >>> state = {"messages": [AIMessage(content="...", tool_calls=[...])]}
        >>> route_after_agent(state)
        "tools"

        >>> state = {"messages": [AIMessage(content="...")]}
        >>> route_after_agent(state)
        "finalize"
    """
    messages = state.get("messages", [])

    if has_pending_tool_calls(messages):
        logger.debug("Agent has pending tool calls, routing to tools")
        return "tools"

    logger.debug("No pending tool calls, routing to finalize")
    return "finalize"


def route_after_prompt_continue(state: AgentState) -> str:
    """
    Route after prompt_continue node: restart conversation or end.

    This routing function determines if the conversation should continue
    with a new query or end.

    Args:
        state: Current graph state with continue_conversation flag

    Returns:
        "reset_for_next_turn" if conversation continues,
        "END" otherwise

    Example:
        >>> state = {"continue_conversation": True}
        >>> route_after_prompt_continue(state)
        "reset_for_next_turn"

        >>> state = {"continue_conversation": False}
        >>> route_after_prompt_continue(state)
        "END"
    """
    continue_conv = state.get("continue_conversation", False)

    if continue_conv:
        logger.debug("Continuing conversation, routing to reset state")
        return "reset_for_next_turn"
    else:
        logger.debug("Ending conversation")
        return "END"
