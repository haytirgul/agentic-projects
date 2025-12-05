"""Prompt templates for the code RAG agent.

This package contains all prompt templates used by the agent nodes.
Following the RAG_agent pattern, prompts are isolated here with message
builder functions for easy LLM integration.
"""

from prompts.router_prompts import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT_TEMPLATE,
    SAMPLE_ROUTER_DECISIONS,
    get_router_prompts,
    build_router_messages,
)
from prompts.synthesizer_prompts import (
    SYNTHESIZER_SYSTEM_PROMPT,
    SYNTHESIZER_USER_PROMPT_TEMPLATE,
    SAMPLE_SYNTHESIZED_ANSWERS,
    get_synthesizer_prompts,
    build_synthesizer_messages,
)

__all__ = [
    "ROUTER_SYSTEM_PROMPT",
    "ROUTER_USER_PROMPT_TEMPLATE",
    "SAMPLE_ROUTER_DECISIONS",
    "get_router_prompts",
    "build_router_messages",
    "SYNTHESIZER_SYSTEM_PROMPT",
    "SYNTHESIZER_USER_PROMPT_TEMPLATE",
    "SAMPLE_SYNTHESIZED_ANSWERS",
    "get_synthesizer_prompts",
    "build_synthesizer_messages",
]
