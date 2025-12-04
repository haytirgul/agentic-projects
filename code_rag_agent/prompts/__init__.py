"""Prompt templates for the code RAG agent.

This package contains all prompt templates used by the agent nodes.
"""

from prompts.router_prompts import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT_TEMPLATE,
    SAMPLE_ROUTER_DECISIONS,
    get_router_prompts
)

__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "ROUTER_USER_PROMPT_TEMPLATE",
    "SAMPLE_ROUTER_DECISIONS",
    "get_router_prompts",
]
