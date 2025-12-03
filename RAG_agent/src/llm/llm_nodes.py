"""
LangGraph node implementations for LLM operations.

This module provides LangGraph-compatible nodes for:
1. Tool-calling agent nodes (iterative tool execution)
2. Tool executor nodes (execute tool calls)
3. Structured output nodes (terminal nodes that produce typed responses)

Design Philosophy:
- Each node has a single responsibility
- Nodes return state updates compatible with LangGraph reducers
- Clean separation between tool-calling and structured output phases
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from src.llm.workers import invoke_with_tools, invoke_structured, execute_tool_calls

__all__ = [
    "tool_agent_node",
    "tool_executor_node",
    "structured_output_node",
    "simple_agent_node",
]

logger = logging.getLogger(__name__)


def tool_agent_node(
    state: dict[str, Any],
    tools: List[BaseTool],
    llm: BaseChatModel,
    messages_key: str = "messages",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    LangGraph node that invokes LLM with tools (single step).

    This node performs ONE LLM invocation. Use with conditional edges
    to loop back through tool execution until no more tool calls.

    Tool-calling nodes never stream since they produce structured tool calls,
    not user-facing content. Use user_output_node for streaming final responses.

    Args:
        state: LangGraph state dict containing messages
        tools: Tools available to the agent
        llm: LLM instance to use for this invocation
        messages_key: Key for messages in state (default: "messages")
        temperature: Optional temperature override for this invocation
        max_tokens: Optional max_tokens override for this invocation

    Returns:
        State update with new AI message appended to messages

    Example:
        >>> # Using default LLM:
        >>> graph.add_node("agent", lambda s: tool_agent_node(s, tools, llm=my_llm))
        >>>
        >>> # Using custom LLM per node:
        >>> custom_llm = ChatGoogleGenerativeAI(model="gemini-pro", ...)
        >>> graph.add_node("agent", partial(tool_agent_node, tools=tools, llm=custom_llm))
    """
    messages = state.get(messages_key, [])
    if not messages:
        raise ValueError(f"No messages found in state['{messages_key}']")

    response = invoke_with_tools(
        messages=list(messages),
        tools=tools,
        llm=llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return {messages_key: [response]}


def simple_agent_node(
    state: dict[str, Any],
    llm: BaseChatModel,
    messages_key: str = "messages",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    LangGraph node that invokes LLM without tools (single step).

    This node performs ONE LLM invocation without tool-calling capability.
    Use this when you don't need tools and just want a direct LLM response.

    Args:
        state: LangGraph state dict containing messages
        llm: LLM instance to use for this invocation
        messages_key: Key for messages in state (default: "messages")
        temperature: Optional temperature override for this invocation
        max_tokens: Optional max_tokens override for this invocation

    Returns:
        State update with new AI message appended to messages

    Example:
        >>> graph.add_node("agent", partial(simple_agent_node, llm=my_llm))
    """
    from src.llm.workers import _bind_llm_params

    messages = state.get(messages_key, [])
    if not messages:
        raise ValueError(f"No messages found in state['{messages_key}']")

    # Bind parameters if provided
    bound_llm = _bind_llm_params(llm, temperature, max_tokens)

    response = bound_llm.invoke(list(messages))

    return {messages_key: [response]}


def tool_executor_node(
    state: dict[str, Any],
    tools: List[BaseTool],
    messages_key: str = "messages",
) -> dict[str, Any]:
    """
    LangGraph node that executes pending tool calls.

    This node finds the last AI message, executes its tool calls,
    and returns the tool results to be appended to messages.

    Args:
        state: LangGraph state dict containing messages
        tools: Available tools
        messages_key: Key for messages in state (default: "messages")

    Returns:
        State update with tool result messages appended

    Example:
        >>> graph.add_node("tools", lambda s: tool_executor_node(s, tools))
        >>> graph.add_edge("tools", "agent")  # Loop back to agent
    """
    messages = state.get(messages_key, [])
    if not messages:
        raise ValueError(f"No messages found in state['{messages_key}']")

    # Find last AI message with tool calls
    last_ai_msg = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            last_ai_msg = msg
            break

    if not last_ai_msg:
        logger.warning("No AI message with tool calls found")
        return {messages_key: []}

    tool_results = execute_tool_calls(last_ai_msg, tools)

    return {messages_key: tool_results}


def structured_output_node(
    state: dict[str, Any],
    output_schema: Type[BaseModel],
    llm: BaseChatModel,
    messages_key: str = "messages",
    output_key: str = "result",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dict[str, Any]:
    """
    LangGraph node that produces structured output (terminal node).

    This should be the final node after tool-calling is complete.
    It invokes the LLM with structured output to produce a typed response.

    Args:
        state: LangGraph state dict containing messages
        output_schema: Pydantic model for the response
        llm: LLM instance to use for this invocation
        messages_key: Key for messages in state (default: "messages")
        output_key: Key to store result in state (default: "result")
        temperature: Optional temperature override for this invocation
        max_tokens: Optional max_tokens override for this invocation

    Returns:
        State update with parsed structured output

    Example:
        >>> # Using default LLM:
        >>> graph.add_node("finalize", lambda s: structured_output_node(s, ResultSchema, llm=my_llm))
        >>>
        >>> # Using custom LLM per node:
        >>> custom_llm = ChatGoogleGenerativeAI(model="gemini-pro", ...)
        >>> graph.add_node("finalize", partial(structured_output_node, output_schema=ResultSchema, llm=custom_llm))
    """
    messages = state.get(messages_key, [])
    if not messages:
        raise ValueError(f"No messages found in state['{messages_key}']")

    result = invoke_structured(
        messages=list(messages),
        output_schema=output_schema,
        llm=llm,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return {output_key: result}
