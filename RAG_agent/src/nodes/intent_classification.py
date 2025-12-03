"""
Intent classification node for LangGraph agent.

PERFORMANCE OPTIMIZED: Single-node implementation with structured output and tool support.

Benefits:
- Single LLM call instead of multiple steps
- Supports tool calls (ask_user) for clarification
- Maintains caching functionality
- Simplified graph routing

Implementation:
- Structured output + tools in one LLM call (3-5s)
- Handles tool calls internally via loop
- Returns IntentClassification directly
"""

from __future__ import annotations

import logging
import hashlib
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from models.intent_classification import ParsedIntentClassification
from src.graph.state import AgentState
from prompts.intent_classification import INTENT_CLASSIFICATION_SYSTEM_PROMPT
from src.llm import get_cached_llm
from src.llm.workers import invoke_structured
from settings import LEVEL_TO_MODEL

__all__ = [
    "intent_classification_node",
]

logger = logging.getLogger(__name__)

# Intent classification cache (in-memory with TTL)
_intent_cache: dict[str, dict[str, Any]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_SIZE = 1000  # Prevent unbounded growth

# Model configuration
INTENT_MODEL_LEVEL = "fast"  # Balance between speed and accuracy
INTENT_TEMPERATURE = 0.1  # Low temperature for consistent classification
INTENT_MAX_TOKENS = 8192
MAX_TOOL_ITERATIONS = 3  # Prevent infinite clarification loops


def _get_cached_intent(cleaned_request: str, conversation_context: str) -> Optional[ParsedIntentClassification]:
    """Check if intent is cached for this query."""
    cache_key = hashlib.md5(f"{cleaned_request}:{conversation_context}".encode()).hexdigest()

    if cache_key in _intent_cache:
        cached = _intent_cache[cache_key]
        age = time.time() - cached['timestamp']

        if age < CACHE_TTL_SECONDS:
            logger.info(f"Intent cache HIT (age: {age:.1f}s) for query: {cleaned_request[:50]}...")
            return cached['intent']
        else:
            # Expired, remove it
            del _intent_cache[cache_key]
            logger.debug(f"Intent cache entry expired for query: {cleaned_request[:50]}...")

    return None


def _set_intent_cache(cleaned_request: str, conversation_context: str, intent: ParsedIntentClassification):
    """Store intent in cache with TTL."""
    cache_key = hashlib.md5(f"{cleaned_request}:{conversation_context}".encode()).hexdigest()

    _intent_cache[cache_key] = {
        'intent': intent,
        'timestamp': time.time()
    }

    # Evict oldest entries if cache is too large
    if len(_intent_cache) > MAX_CACHE_SIZE:
        oldest_key = min(_intent_cache.items(), key=lambda x: x[1]['timestamp'])[0]
        del _intent_cache[oldest_key]
        logger.debug(f"Evicted oldest intent cache entry (size: {len(_intent_cache)})")


def intent_classification_node(state: AgentState) -> dict[str, Any]:
    """
    Parse user input and classify intent (merged preprocessing + intent classification).

    OPTIMIZED: Combines preprocessing + intent classification into single node for better performance.

    This node:
    1. Parses raw user input (extracts code, cleans request, security analysis)
    2. Checks cache for identical queries (1-hour TTL)
    3. Builds intent classification messages
    4. Calls LLM with structured output schema
    5. Returns structured IntentClassification with parsed components
    6. Caches result for future queries

    Args:
        state: Current graph state (must contain user_input)

    Returns:
        State update with:
        - cleaned_request: Cleaned text
        - code_snippets: Extracted code blocks
        - extracted_data: Structured data (errors, configs)
        - query_type: Type of query
        - security_analysis: Security analysis results
        - intent_result: IntentClassification
        - messages: Updated message history
        - final_response: Reset to None

    Example:
        >>> state = {"user_input": "Hey! How do I add persistence to my agent?"}
        >>> result = intent_classification_node(state)
        >>> print(result["cleaned_request"])
        'How do I add persistence to my agent?'
        >>> print(result["intent_result"].intent_type)
        'implementation_guide'
    """
    # Get raw user input
    user_input = state.get("user_input")
    if not user_input:
        raise ValueError("user_input is required in state")

    try:
        # Get conversation history
        conversation_history = state.get("conversation_memory")

        # Check cache first (using raw user input for cache key)
        conversation_context = conversation_history.current_topic if conversation_history else ""
        cached_intent = _get_cached_intent(user_input, conversation_context)
        if cached_intent:
            # Extract parsed fields from cached result (assuming we cached ParsedIntentClassification)
            return {
                "cleaned_request": cached_intent.cleaned_request[0] if cached_intent.cleaned_request else user_input,
                "code_snippets": cached_intent.code_snippets or [],
                "extracted_data": cached_intent.data,
                "query_type": cached_intent.query_type,
                "security_analysis": cached_intent.security_analysis or {},
                "intent_result": cached_intent,
            }

        logger.info(f"Parsing and classifying in single LLM call: {user_input[:100]}...")

        # Build messages for COMBINED parsing + classification
        messages = [
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=f"User Input: {user_input}")
        ]

        # Add conversation history if available
        if conversation_history and conversation_history.turns:
            history_text = "\n\nConversation History (recent turns):\n"
            for turn in conversation_history.turns[-3:]:  # Last 3 turns
                history_text += f"User: {turn.user_query}\nAssistant: {turn.assistant_response}\n"
            messages.append(HumanMessage(content=history_text))

        # SINGLE LLM CALL: Get LLM and invoke with combined model
        model_name = LEVEL_TO_MODEL[INTENT_MODEL_LEVEL]
        llm = get_cached_llm(model_name)
        logger.debug(f"Using {model_name} ({INTENT_MODEL_LEVEL}) for combined parsing + classification")

        # Use the combined model that includes BOTH parsing and classification fields
        result = invoke_structured(
            messages=list(messages),
            output_schema=ParsedIntentClassification,
            llm=llm,
            temperature=INTENT_TEMPERATURE,
            max_tokens=INTENT_MAX_TOKENS,
        )

        if isinstance(result, ParsedIntentClassification):
            # Extract components
            request_text = result.cleaned_request[0] if result.cleaned_request else user_input
            code_snippets = result.code_snippets or []
            extracted_data = result.data
            query_type = result.query_type
            security_analysis = result.security_analysis or {}

            # Log security analysis if present
            suspicious_patterns = security_analysis.get("suspicious_patterns", [])
            risk_level = security_analysis.get("risk_level", "low")
            if suspicious_patterns:
                logger.warning(
                    f"Security analysis detected {len(suspicious_patterns)} suspicious patterns: "
                    f"{', '.join(suspicious_patterns)} (risk: {risk_level})"
                )

            logger.info(
                f"✓ Parsed + Classified in single call: type={result.intent_type}, "
                f"framework={result.framework}, "
                f"snippets={len(code_snippets)}, "
                f"risk={risk_level}"
            )

            # Cache the result (cache the full combined model)
            _set_intent_cache(user_input, conversation_context, result)

            return {
                "cleaned_request": request_text,
                "code_snippets": code_snippets,
                "extracted_data": extracted_data,
                "query_type": query_type,
                "security_analysis": security_analysis,
                "intent_result": result,  # Full combined model
                "messages": messages + [AIMessage(content=str(result))],
                # Reset per-turn state
                "final_response": None,
            }

        # Complete failure
        raise RuntimeError("Failed to extract ParsedIntentClassification")

    except Exception as e:
        logger.error(f"Intent classification failed: {e}", exc_info=True)
        raise RuntimeError(f"Intent classification failed: {e}") from e
