"""Prompts module for LLM interactions.

This module contains all LLM prompts used by the Code RAG Agent,
including router prompts for query decomposition, synthesis prompts
for answer generation, and intent prompts for query classification.

Author: Hay Hoffman
Version: 1.2
"""

from prompts.intent_prompt import (
    INTENT_SYSTEM_PROMPT,
    INTENT_USER_PROMPT_TEMPLATE,
    build_intent_prompt,
)
from prompts.router_prompt import (
    ROUTER_PROMPT,
    build_router_prompt,
    generate_codebase_tree,
)
from prompts.synthesis_prompt import (
    build_synthesis_messages,
    build_system_prompt,
    build_user_prompt,
)

__all__ = [
    # Intent classification prompts
    "INTENT_SYSTEM_PROMPT",
    "INTENT_USER_PROMPT_TEMPLATE",
    "build_intent_prompt",
    # Router prompts
    "ROUTER_PROMPT",
    "build_router_prompt",
    "generate_codebase_tree",
    # Synthesis prompts
    "build_synthesis_messages",
    "build_system_prompt",
    "build_user_prompt",
]
