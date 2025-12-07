"""Input preprocessor node for query cleaning and intent classification.

This node preprocesses user queries before routing using a hybrid approach:
1. Fast path: Regex patterns for explicit history queries and greetings (<1ms)
2. LLM fallback: Intent classification for ambiguous queries

The LLM fallback uses structured output to:
- Classify intent (history_request, follow_up, new_question, general_question)
- Resolve pronoun references to previous context
- Provide reasoning for classification decisions

Author: Hay Hoffman
Version: 1.3
"""

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from models.intent import QueryIntent
from prompts.intent_prompt import INTENT_SYSTEM_PROMPT, build_intent_prompt
from settings import INTENT_MAX_TOKENS, INTENT_MODEL, INTENT_TEMPERATURE

from src.agent.state import AgentState
from src.llm import get_cached_llm, invoke_structured

__all__ = ["input_preprocessor_node"]

logger = logging.getLogger(__name__)

# ============================================================================
# REGEX PATTERNS (Fast Path)
# ============================================================================

# Patterns for detecting explicit history-related queries
HISTORY_PATTERNS = [
    r"what did (i|we|you) (ask|say|discuss|talk about)",
    r"(earlier|previous|last|first) (question|answer|query|response)",
    r"summarize (our|the|this) conversation",
    r"what was my (first|last|previous) question",
    r"what have (we|i) (discussed|talked about|asked)",
    r"remind me what (i|we) (asked|said)",
    r"(show|list|give) me (the|our|my) (conversation|chat) history",
    r"what (questions|queries) have i asked",
]

# Patterns for detecting greetings and simple responses (general_question fast path)
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good (morning|afternoon|evening))[\s!.,?]*$",
    r"^(thanks|thank you|thx|ty)[\s!.,]*$",
    r"^(bye|goodbye|see you|later)[\s!.,]*$",
    r"^(ok|okay|sure|got it|understood|alright)[\s!.,]*$",
    r"^(yes|no|yep|nope|yeah|nah)[\s!.,]*$",
]

# Compiled patterns for performance
_COMPILED_HISTORY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in HISTORY_PATTERNS
]

_COMPILED_GREETING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in GREETING_PATTERNS
]

# Filler words to remove from queries
FILLER_WORDS = {
    "please",
    "could you",
    "can you",
    "would you",
    "i want to",
    "i need to",
    "i would like to",
    "help me",
    "tell me",
    "show me",
    "explain to me",
    "just",
    "actually",
    "basically",
    "quickly",
    "maybe",
    "perhaps",
    "kind of",
    "sort of",
}


# ============================================================================
# FAST PATH FUNCTIONS
# ============================================================================


def _is_explicit_history_query(query: str) -> bool:
    """Check if query explicitly asks about conversation history.

    Uses regex patterns for fast matching (<1ms).

    Args:
        query: User query string

    Returns:
        True if query matches any explicit history pattern
    """
    for pattern in _COMPILED_HISTORY_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _is_greeting_or_simple_response(query: str) -> bool:
    """Check if query is a greeting or simple response (no retrieval needed).

    Uses regex patterns for fast matching (<1ms).

    Args:
        query: User query string

    Returns:
        True if query is a greeting, thanks, or simple acknowledgment
    """
    for pattern in _COMPILED_GREETING_PATTERNS:
        if pattern.search(query.strip()):
            return True
    return False


def _clean_query(query: str) -> str:
    """Clean query by removing filler words and normalizing whitespace.

    Args:
        query: Raw user query

    Returns:
        Cleaned query string
    """
    cleaned = query.strip()

    # Remove common filler phrases (case-insensitive)
    for filler in FILLER_WORDS:
        pattern = re.compile(rf"\b{re.escape(filler)}\b", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove leading punctuation that might remain
    cleaned = re.sub(r"^[,.\s]+", "", cleaned)

    return cleaned if cleaned else query  # Fallback to original if empty


def _format_history_response(conversation_history: list[dict]) -> str:
    """Format conversation history into a readable response.

    Args:
        conversation_history: list of conversation turns

    Returns:
        Formatted string summarizing the conversation
    """
    if not conversation_history:
        return "No previous conversation history found."

    response_parts = ["Here's a summary of our conversation:\n"]

    for i, turn in enumerate(conversation_history, 1):
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        # Truncate long answers for summary
        answer_preview = answer[:200] + "..." if len(answer) > 200 else answer

        response_parts.append(f"**Turn {i}:**")
        response_parts.append(f"- Question: {query}")
        response_parts.append(f"- Answer: {answer_preview}\n")

    return "\n".join(response_parts)


# ============================================================================
# LLM INTENT CLASSIFICATION (Fallback)
# ============================================================================


def _classify_intent_with_llm(
    user_query: str,
    conversation_history: list[dict],
) -> QueryIntent:
    """Classify query intent using LLM with structured output.

    Uses INTENT_MODEL (gemini-2.5-flash-lite) for low latency.

    Args:
        user_query: Current user query
        conversation_history: list of previous Q&A turns

    Returns:
        QueryIntent with classification and resolved query

    Raises:
        RuntimeError: If LLM classification fails
    """
    logger.info("Using LLM for intent classification")

    # Build messages
    user_prompt = build_intent_prompt(user_query, conversation_history)
    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # Get cached LLM and invoke with structured output
    llm = get_cached_llm(INTENT_MODEL)
    intent = invoke_structured(
        messages=messages,
        output_schema=QueryIntent,
        llm=llm,
        temperature=INTENT_TEMPERATURE,
        max_tokens=INTENT_MAX_TOKENS,
    )

    logger.info(
        f"LLM classified intent: {intent.intent} "
        f"(references_previous={intent.references_previous})"
    )
    return intent


# ============================================================================
# MAIN NODE FUNCTION
# ============================================================================


def input_preprocessor_node(state: AgentState) -> dict[str, Any]:
    """Preprocess user input with hybrid intent classification.

    This node uses a two-stage approach:
    1. Fast path: Regex matching for explicit history queries (<1ms)
    2. LLM fallback: Intent classification when history exists (~500ms)

    The LLM is only invoked when:
    - Regex didn't match an explicit history query
    - Conversation history is not empty (no point classifying on first query)

    Args:
        state: Current agent state containing user_query

    Returns:
        Updated state with:
            - cleaned_query: Preprocessed/resolved query string
            - is_history_query: True if asking about history
            - is_follow_up: True if query references previous context
            - final_answer: Set if history query (skip retrieval)

    Raises:
        ValueError: If user_query is missing from state

    Example:
        >>> state = {"user_query": "Tell me more about that", "conversation_history": [...]}
        >>> result = input_preprocessor_node(state)
        >>> result["is_follow_up"]
        True
        >>> result["cleaned_query"]
        "Tell me more about BM25 tokenization"  # Resolved reference
    """
    user_query = state.get("user_query")

    if not user_query:
        raise ValueError("user_query is required for input preprocessor")

    logger.info(f"Preprocessing query: {user_query[:100]}...")

    conversation_history = state.get("conversation_history", []) or []

    # -------------------------------------------------------------------------
    # STAGE 1: Fast path - Regex matching for explicit history queries
    # -------------------------------------------------------------------------
    if _is_explicit_history_query(user_query):
        logger.info("[FAST PATH] Detected explicit history query via regex")
        history_response = _format_history_response(conversation_history)

        return {
            "cleaned_query": user_query,
            "is_history_query": True,
            "is_follow_up": False,
            "needs_retrieval": False,
            "final_answer": history_response,
            "citations": [],
        }

    # -------------------------------------------------------------------------
    # STAGE 1b: Fast path - Greetings and simple responses (no retrieval)
    # -------------------------------------------------------------------------
    if _is_greeting_or_simple_response(user_query):
        logger.info("[FAST PATH] Detected greeting/simple response via regex")

        return {
            "cleaned_query": user_query,
            "is_history_query": False,
            "is_follow_up": False,
            "is_general_question": True,
            "needs_retrieval": False,
        }

    # -------------------------------------------------------------------------
    # STAGE 2: LLM fallback - Only if history exists
    # -------------------------------------------------------------------------
    if conversation_history:
        logger.info("[LLM PATH] History exists, classifying intent with LLM")

        try:
            intent = _classify_intent_with_llm(user_query, conversation_history)

            if intent.intent == "history_request":
                # LLM detected implicit history request
                logger.info("LLM classified as history_request")
                history_response = _format_history_response(conversation_history)

                return {
                    "cleaned_query": user_query,
                    "is_history_query": True,
                    "is_follow_up": False,
                    "needs_retrieval": False,
                    "final_answer": history_response,
                    "citations": [],
                }

            elif intent.intent == "follow_up":
                # Query references previous context AND history is sufficient
                logger.info(
                    f"LLM classified as follow_up (history sufficient), "
                    f"resolved: {intent.resolved_query[:100]}..."
                )

                return {
                    "cleaned_query": intent.resolved_query,
                    "is_history_query": False,
                    "is_follow_up": True,
                    "needs_retrieval": False,
                }

            elif intent.intent == "follow_up_with_retrieval":
                # Query references previous context BUT needs fresh retrieval
                logger.info(
                    f"LLM classified as follow_up_with_retrieval (new info needed), "
                    f"resolved: {intent.resolved_query[:100]}..."
                )

                return {
                    "cleaned_query": intent.resolved_query,
                    "is_history_query": False,
                    "is_follow_up": True,
                    "needs_retrieval": True,
                }

            elif intent.intent == "general_question":
                # General question - can be answered without retrieval
                logger.info("LLM classified as general_question (no retrieval needed)")

                return {
                    "cleaned_query": intent.resolved_query,
                    "is_history_query": False,
                    "is_follow_up": False,
                    "is_general_question": True,
                    "needs_retrieval": False,
                }

            elif intent.intent == "out_of_scope":
                # Query contains non-HTTPX code - politely decline
                logger.info("LLM classified as out_of_scope (non-HTTPX code detected)")

                out_of_scope_response = (
                    "I can only help with questions about the **HTTPX Python library**.\n\n"
                    "Your query appears to involve code that doesn't use HTTPX "
                    "(e.g., custom clients, requests, aiohttp, or other HTTP libraries).\n\n"
                    "**How I can help:**\n"
                    "- Questions about `httpx.Client`, `httpx.AsyncClient`, and their methods\n"
                    "- Understanding HTTPX internals, connection pooling, timeouts\n"
                    "- Debugging code that uses HTTPX classes and functions\n\n"
                    "Please rephrase your question to be about HTTPX specifically."
                )

                return {
                    "cleaned_query": user_query,
                    "is_history_query": False,
                    "is_follow_up": False,
                    "is_out_of_scope": True,
                    "needs_retrieval": False,
                    "final_answer": out_of_scope_response,
                    "citations": [],
                }

            else:  # new_question
                # Independent query about httpx - needs retrieval
                logger.info("LLM classified as new_question (httpx-specific)")
                cleaned_query = _clean_query(intent.resolved_query)

                return {
                    "cleaned_query": cleaned_query,
                    "is_history_query": False,
                    "is_follow_up": False,
                    "is_general_question": False,
                    "needs_retrieval": True,
                }

        except RuntimeError as e:
            # LLM failed - fall back to simple cleaning (assume retrieval needed)
            logger.warning(f"LLM classification failed, falling back: {e}")
            cleaned_query = _clean_query(user_query)

            return {
                "cleaned_query": cleaned_query,
                "is_history_query": False,
                "is_follow_up": False,
                "needs_retrieval": True,
            }

    # -------------------------------------------------------------------------
    # STAGE 3: No history - Use LLM to detect general vs httpx-specific questions
    # -------------------------------------------------------------------------
    logger.info("[FIRST QUERY] No history, classifying intent with LLM")

    try:
        intent = _classify_intent_with_llm(user_query, [])

        if intent.intent == "general_question":
            logger.info("First query classified as general_question (no retrieval)")

            return {
                "cleaned_query": intent.resolved_query,
                "is_history_query": False,
                "is_follow_up": False,
                "is_general_question": True,
                "needs_retrieval": False,
            }

        elif intent.intent == "out_of_scope":
            # Query contains non-HTTPX code - politely decline
            logger.info("First query classified as out_of_scope (non-HTTPX code detected)")

            out_of_scope_response = (
                "I can only help with questions about the **HTTPX Python library**.\n\n"
                "Your query appears to involve code that doesn't use HTTPX "
                "(e.g., custom clients, requests, aiohttp, or other HTTP libraries).\n\n"
                "**How I can help:**\n"
                "- Questions about `httpx.Client`, `httpx.AsyncClient`, and their methods\n"
                "- Understanding HTTPX internals, connection pooling, timeouts\n"
                "- Debugging code that uses HTTPX classes and functions\n\n"
                "Please rephrase your question to be about HTTPX specifically."
            )

            return {
                "cleaned_query": user_query,
                "is_history_query": False,
                "is_follow_up": False,
                "is_out_of_scope": True,
                "needs_retrieval": False,
                "final_answer": out_of_scope_response,
                "citations": [],
            }

        else:  # new_question (most likely for first query)
            logger.info("First query classified as new_question (httpx-specific)")
            cleaned_query = _clean_query(intent.resolved_query)

            return {
                "cleaned_query": cleaned_query,
                "is_history_query": False,
                "is_follow_up": False,
                "is_general_question": False,
                "needs_retrieval": True,
            }

    except RuntimeError as e:
        # LLM failed - fall back to simple cleaning (assume retrieval needed)
        logger.warning(f"LLM classification failed for first query, assuming retrieval: {e}")
        cleaned_query = _clean_query(user_query)

        return {
            "cleaned_query": cleaned_query,
            "is_history_query": False,
            "is_follow_up": False,
            "is_general_question": False,
            "needs_retrieval": True,
        }
