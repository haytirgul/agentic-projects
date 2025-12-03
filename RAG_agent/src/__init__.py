"""
Agent source module.

Provides LLM factories, LangGraph node builders, and agent utilities.

Two-Phase Pattern for Tool Calling + Structured Output:

    1. Tool-calling phase: Use `tool_agent_node` and `tool_executor_node`
       in a loop until `has_pending_tool_calls()` returns False.
    2. Structured output phase: Use `structured_output_node` to produce
       typed responses.

Example graph structure:
    START -> agent -> (conditional) -> tools -> agent
                   -> finalize -> END
"""

from src.llm import create_llm_with_config
from src.llm.workers import (
    # Core invocation functions
    invoke_structured,
    invoke_with_tools,
    # Tool execution
    execute_tool_calls,
    # Routing helpers
    has_pending_tool_calls,
)

# Import node factories lazily to avoid circular imports
def _get_node_factories():
    """Lazy import to avoid circular dependencies."""
    from src.llm.llm_nodes import (
        structured_output_node,
        tool_agent_node,
        tool_executor_node,
    )
    return tool_agent_node, tool_executor_node, structured_output_node

# For direct imports (e.g., from src import tool_agent_node)
def __getattr__(name):
    """Lazy attribute access for node factories."""
    if name in ("tool_agent_node", "tool_executor_node", "structured_output_node"):
        from src.llm.llm_nodes import (
            structured_output_node as _so,
            tool_agent_node as _ta,
            tool_executor_node as _te,
        )
        mapping = {
            "tool_agent_node": _ta,
            "tool_executor_node": _te,
            "structured_output_node": _so,
        }
        return mapping[name]
    raise AttributeError(f"module 'src' has no attribute '{name}'")

__all__ = [
    # LLM factories
    "create_llm_with_config",
    # Core invocation
    "invoke_with_tools",
    "invoke_structured",
    # Tool execution
    "execute_tool_calls",
    # Routing
    "has_pending_tool_calls",
    # LangGraph node factories
    "tool_agent_node",
    "tool_executor_node",
    "structured_output_node",
]
