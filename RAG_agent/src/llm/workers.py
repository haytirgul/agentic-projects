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

import json
import logging
from typing import Any, List, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel


__all__ = [
    "invoke_with_tools",
    "invoke_structured",
    "execute_tool_calls",
    "has_pending_tool_calls",
]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# =============================================================================
# Helper Functions
# =============================================================================


def _bind_llm_params(llm: Any, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Any:  # type: ignore[return]
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


def invoke_with_tools(
    messages: List[BaseMessage],
    tools: List[BaseTool],
    llm: BaseChatModel,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> AIMessage:
    """
    Invoke LLM with tools bound (single invocation, no loop).

    This performs ONE LLM call with tools available. The caller is responsible
    for handling the tool-calling loop (typically via LangGraph edges).

    Tool-calling invocations never stream since they produce structured tool calls,
    not user-facing content.

    Args:
        messages: Conversation history
        tools: Tools to bind to the LLM
        llm: Pre-configured LLM instance (required). Tools will be bound to this LLM.
        temperature: Optional temperature to bind to LLM for this invocation
        max_tokens: Optional max_tokens to bind to LLM for this invocation
        **kwargs: Additional arguments passed to LLM invoke

    Returns:
        AIMessage from the LLM (may contain tool_calls)

    Raises:
        RuntimeError: If LLM invocation fails

    Example:
        >>> # Using custom LLM with params
        >>> llm = get_cached_llm("gemini-2.5-flash")
        >>> response = invoke_with_tools(messages, tools, llm=llm, temperature=0.1, max_tokens=4096)
    """
    if not tools:
        raise ValueError("At least one tool must be provided")

    if llm is None:
        raise ValueError("LLM instance must be provided explicitly")

    try:
        # Bind params first, then bind tools (order matters for RunnableBinding compatibility)
        llm_with_params = _bind_llm_params(llm, temperature, max_tokens)
        llm_with_tools = llm_with_params.bind_tools(tools)

        logger.debug(f"Invoking provided LLM with {len(tools)} tool(s)")

        # Always use invoke (no streaming for tool calls)
        response = llm_with_tools.invoke(messages, **kwargs)

        tool_call_count = len(getattr(response, "tool_calls", []) or [])
        logger.info(f"LLM response received (tool_calls: {tool_call_count})")

        return response  # type: ignore[return-value]
    except Exception as e:
        logger.error(f"Tool-calling invocation failed: {e}")
        raise RuntimeError(f"Failed to invoke LLM with tools: {e}") from e


def invoke_structured(
    messages: List[BaseMessage],
    output_schema: Type[T],
    llm: BaseChatModel,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
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
# Tool Execution
# =============================================================================


def execute_tool_calls(
    ai_message: AIMessage,
    tools: List[BaseTool],
) -> List[ToolMessage]:
    """
    Execute all tool calls from an AI message.

    Args:
        ai_message: AIMessage containing tool_calls
        tools: Available tools to execute

    Returns:
        List of ToolMessage results (one per tool call)

    Example:
        >>> tool_results = execute_tool_calls(response, tools)
        >>> messages.extend(tool_results)
    """
    tool_calls = getattr(ai_message, "tool_calls", []) or []
    if not tool_calls:
        return []

    tool_map = {tool.name: tool for tool in tools}
    results: List[ToolMessage] = []

    for tc in tool_calls:
        tool_name: Optional[str] = (
            tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        )
        tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        tool_id: Optional[str] = (
            tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
        )

        # Skip if tool name is missing
        if not tool_name:
            logger.warning("Tool call missing 'name' field, skipping")
            continue

        # Parse string args if needed
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                pass

        tool = tool_map.get(tool_name)
        if not tool:
            logger.warning(f"Tool '{tool_name}' not found, skipping")
            results.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_id or f"call_{tool_name}",
                )
            )
            continue

        try:
            result = tool.invoke(tool_args)
            logger.debug(f"Tool '{tool_name}' executed successfully")

            results.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id or f"call_{tool_name}",
                )
            )
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}")

            results.append(
                ToolMessage(
                    content=f"Error executing {tool_name}: {e}",
                    tool_call_id=tool_id or f"call_{tool_name}",
                )
            )

    logger.info(f"Executed {len(results)} tool call(s)")
    return results


# =============================================================================
# Conditional Routing Helpers
# =============================================================================


def has_pending_tool_calls(messages: List[BaseMessage]) -> bool:
    """
    Check if the last message has pending tool calls.

    Use this in LangGraph conditional edges to route between
    tool execution and structured output nodes.

    Args:
        messages: Current message history

    Returns:
        True if last message is AIMessage with tool_calls

    Example:
        >>> def route_after_agent(state):
        ...     if has_pending_tool_calls(state["messages"]):
        ...         return "execute_tools"
        ...     return "finalize"
    """
    if not messages:
        return False

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage):
        return False

    tool_calls = getattr(last_msg, "tool_calls", []) or []
    return len(tool_calls) > 0


# =============================================================================
# Private Helpers
# =============================================================================


def _is_valid_schema(schema: Any) -> bool:
    """Check if schema is a valid Pydantic BaseModel subclass."""
    try:
        return isinstance(schema, type) and issubclass(schema, BaseModel)
    except TypeError:
        return False


