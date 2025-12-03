"""
Prompt templates for the opsfleet task.
"""

from prompts.intent_classification import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    build_intent_classification_messages,
)

__all__ = [
    "INTENT_CLASSIFICATION_SYSTEM_PROMPT",
    "build_intent_classification_messages",
]

