"""
LangGraph graph builder for the documentation assistant.

Architecture with hybrid RAG:
    START -> preprocessing -> intent_classification_agent -> intent_classification_tools (loop)
          -> intent_classification_finalize -> hybrid_retrieval -> prepare_messages
          -> agent <-> tools (loop) -> finalize -> END

The agent uses the ask_user tool for clarification instead of graph interrupts,
providing a more natural agentic interaction pattern.

Hybrid RAG retrieval uses BM25 + embeddings + fuzzy matching for accurate document retrieval.
"""

from src.graph.builder import (
    build_agent_graph,
    get_compiled_graph,
)
from src.graph.state import AgentState
from src.graph.routing import route_after_intent_classification, route_after_agent
from src.nodes.agent_response import prepare_agent_messages_node, extract_response_node
from prompts.agent_response import AGENT_SYSTEM_PROMPT

__all__ = [
    # Graph builders
    "build_agent_graph",
    "get_compiled_graph",
    # State schemas
    "AgentState",
    # Nodes
    "prepare_agent_messages_node",
    "extract_response_node",
    # Routing
    "route_after_intent_classification",
    "route_after_agent",
    # Prompts
    "AGENT_SYSTEM_PROMPT",
]
