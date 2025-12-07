"""Security gateway node for validating user inputs.

This node performs ML-based prompt injection detection on all user inputs
before they reach the LLM or RAG pipeline, protecting against:
- Prompt injection attacks
- Jailbreak attempts
- Malicious content

Architecture:
    User Input → Security Gateway → Router → Retrieval → Synthesis → Response
"""

import logging
from typing import Any  # noqa: UP035

from settings import SECURITY_ENABLED
from langchain_core.messages import SystemMessage

from src.agent.state import AgentState

__all__ = ["security_gateway_node"]

logger = logging.getLogger(__name__)


def security_gateway_node(state: AgentState) -> dict[str, Any]:
    """Validate user input for security threats using ML classifier.

    This node is the first node in the graph. It validates all user inputs
    using ProtectAI DeBERTa v3 Base to detect:
    - Prompt injection attempts (e.g., "ignore previous instructions")
    - Jailbreak attempts (e.g., "pretend you are...")
    - Malicious content targeting LLMs

    The validation uses a chunking strategy (sentence-level) to validate
    each piece of the input independently.

    Args:
        state: Current agent state containing user_query

    Returns:
        Updated state with:
            - gateway_passed (bool): True if validation passed
            - gateway_reason (str): Explanation if blocked
            - error (str): Error message if blocked
            - is_blocked (bool): True if request was blocked

    Raises:
        ValueError: If user_query is missing from state
    """
    user_query = state.get("user_query")

    if not user_query:
        raise ValueError("user_query is required for security gateway validation")

    # Skip validation if security is disabled
    if not SECURITY_ENABLED:
        logger.debug("Security gateway: SKIPPED (security disabled)")
        return {
            "gateway_passed": True,
            "gateway_reason": None,
        }

    logger.info(f"Security gateway validating input: {user_query[:100]}...")

    try:
        # Import here to avoid loading transformers unless security is enabled
        from src.security import SecurityValidationError, validate_user_input

        # Validate user input (raises SecurityValidationError if malicious)
        # Uses semantic text splitting to chunk text and batch classifies all chunks
        validate_user_input(text=user_query, raise_on_malicious=True)

        # Validation passed
        logger.info("Security gateway: Input validated as SAFE")

        return {
            "gateway_passed": True,
            "gateway_reason": None,
        }

    except SecurityValidationError as e:
        # Malicious content detected
        logger.warning(
            f"Security gateway BLOCKED input: {e.detected_type} "
            f"(confidence: {e.confidence:.2%})"
        )

        # Build user-facing error message
        error_message = (
            "⚠️ Security Validation Failed\n\n"
            "Your request has been blocked by the security system.\n\n"
            f"Reason: {str(e)}\n\n"
            "If you believe this is a mistake, please rephrase your question "
            "and try again."
        )

        # Block the request by setting error and is_blocked flag
        return {
            "gateway_passed": False,
            "gateway_reason": str(e),
            "error": error_message,
            "is_blocked": True,
            "messages": [
                SystemMessage(
                    content=(
                        "Security gateway blocked this request due to detected "
                        f"security threat: {e.detected_type}"
                    )
                )
            ],
        }

    except Exception as e:
        # Unexpected error during validation
        logger.error(f"Security gateway error: {e}", exc_info=True)

        # Fail-open: log error but allow request to proceed
        # This ensures availability even if security module fails
        logger.warning("Security gateway failed, allowing request to proceed (degraded mode)")

        return {
            "gateway_passed": True,
            "gateway_reason": f"Validation error (degraded mode): {e}",
        }
