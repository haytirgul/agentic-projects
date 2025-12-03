"""
LangGraph nodes for the documentation assistant.

This module contains node functions for the LangGraph agent workflow.
Each node is a standalone function that takes state and returns state updates.

Architecture with security gateway and combined intent classification:
- security_gateway_node: ML-based prompt injection detection (first node)
- intent_classification_node: Combined parsing + intent classification in single LLM call
- user_output_node: Display final response to user with optional streaming
"""

from src.nodes.security_gateway import security_gateway_node
from src.nodes.intent_classification import intent_classification_node
from src.llm.llm_nodes import (
    tool_agent_node,
    tool_executor_node,
    structured_output_node,
)
from src.nodes.user_output import user_output_node

__all__ = [
    # Security
    "security_gateway_node",
    # Intent classification
    "intent_classification_node",
    # LLM operations
    "tool_agent_node",
    "tool_executor_node",
    "structured_output_node",
    # User interaction
    "user_output_node",
]
