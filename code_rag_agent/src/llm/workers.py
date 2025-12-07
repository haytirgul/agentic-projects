"""
LLM workers for LangGraph nodes with clean separation of tool-calling and structured output.

This module provides composable functions for:
1. Tool-calling agent nodes (iterative tool execution)
2. Structured output nodes (terminal nodes that produce typed responses)
3. Utility functions for conditional routing

Design Philosophy:
- Each function has a single responsibility
- Tool calling and structured output are separate phases
- Functions return state updates compatible with LangGraph reducers
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

__all__ = [
    "invoke_structured",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Helper Functions
# =============================================================================


def _bind_llm_params(llm: Any, temperature: float | None = None, max_tokens: int | None = None) -> Any:  # type: ignore[return]
    """
    Bind temperature and max_tokens parameters to an LLM or Runnable instance.

    Args:
        llm: Base LLM instance or Runnable to bind parameters to
        temperature: Optional temperature to bind
        max_tokens: Optional max_tokens to bind (converted to max_output_tokens)

    Returns:
        Runnable LLM instance with parameters bound, or original LLM if no params provided
    """
    bind_kwargs = {}
    if temperature is not None:
        bind_kwargs["temperature"] = temperature
    if max_tokens is not None:
        bind_kwargs["max_output_tokens"] = max_tokens

    if bind_kwargs:
        return llm.bind(**bind_kwargs)
    return llm


# =============================================================================
# Core Invocation Functions
# =============================================================================



def invoke_structured(
    messages: list[BaseMessage],
    output_schema: type[T],
    llm: BaseChatModel,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> T:
    """
    Invoke LLM with structured output schema (no tools).

    This is typically used as the terminal node after tool-calling is complete.
    The LLM will produce output conforming to the Pydantic schema.

    Args:
        messages: Conversation history (including tool results)
        output_schema: Pydantic model class for the response
        llm: Pre-configured LLM instance (required). Structured output will be
            configured on this LLM.
        temperature: Optional temperature to bind to LLM for this invocation
        max_tokens: Optional max_tokens to bind to LLM for this invocation
        **kwargs: Additional arguments passed to LLM invoke

    Returns:
        Instance of output_schema with parsed response

    Raises:
        ValueError: If output_schema is not a valid Pydantic model
        RuntimeError: If structured output parsing fails

    Example:
        >>> # Using default LLM
        >>> result = invoke_structured(messages, AnalysisResult)
        >>>
        >>> # Using custom LLM with params
        >>> llm = get_cached_llm("gemini-2.5-flash-lite")
        >>> result = invoke_structured(messages, AnalysisResult, llm=llm, temperature=0.0, max_tokens=2048)
    """
    if not _is_valid_schema(output_schema):
        raise ValueError(
            f"output_schema must be a Pydantic BaseModel subclass, got {type(output_schema)}"
        )

    if llm is None:
        raise ValueError("LLM instance must be provided explicitly")

    try:
        # Bind params if provided
        bound_llm = _bind_llm_params(llm, temperature, max_tokens)

        llm_structured = bound_llm.with_structured_output(output_schema)
        logger.debug(f"Invoking provided LLM for structured output: {output_schema.__name__}")

        # Fix for Gemini models: ensure messages end with user/human role
        # Gemini API requires that structured output calls end with user role
        if messages[-1].type != "human" and messages[-1].type != "system":
            messages.append(HumanMessage(content="Please provide the structured output based on the above conversation."))
        result = llm_structured.invoke(messages, **kwargs)

        if not isinstance(result, output_schema):
            raise ValueError(
                f"Expected {output_schema.__name__}, got {type(result).__name__}"
            )

        logger.info(f"Structured output parsed: {output_schema.__name__}")

        return result
    except Exception as e:
        logger.error(f"Structured output invocation failed: {e}")
        raise RuntimeError(
            f"Failed to get structured output for {output_schema.__name__}: {e}"
        ) from e

# =============================================================================
# Private Helpers
# =============================================================================


def _is_valid_schema(schema: Any) -> bool:
    """Check if schema is a valid Pydantic BaseModel subclass."""
    try:
        return isinstance(schema, type) and issubclass(schema, BaseModel)
    except TypeError:
        return False


