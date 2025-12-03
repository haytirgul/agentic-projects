"""
LLM module for the OpsFleet task agent.

This module provides centralized LLM management with caching, tool binding,
and structured output operations.
"""

from .llm import (
    create_llm_with_config,
    get_cached_llm,
    initialize_llm_cache,
)
from .workers import (
    invoke_with_tools,
    invoke_structured,
    execute_tool_calls,
    has_pending_tool_calls,
)

__all__ = [
    # LLM factory functions
    "create_llm_with_config",
    "get_cached_llm",
    "initialize_llm_cache",
    # Core operations
    "invoke_with_tools",
    "invoke_structured",
    "execute_tool_calls",
    "has_pending_tool_calls",
]
