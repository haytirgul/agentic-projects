"""LangGraph builder for the Code RAG Agent (v1.2).

This module constructs the agent graph with:
- Security gateway (prompt injection detection)
- Input preprocessor (query cleaning, history detection)
- Router (query decomposition with fast path optimization)
- Hybrid retrieval (weighted RRF with code-aware BM25 + context expansion)
- Synthesis (streaming answer generation with citations)
- Conversation memory (follow-up question support)

Graph Flow:
    START → security_gateway → [SECURITY CHECK]
          → END (if blocked)
          → input_preprocessor → [HISTORY CHECK]
          → conversation_memory (if history query)
          → router → retrieval → synthesis → conversation_memory
          → security_gateway (loop) OR END

Author: Hay Hoffman
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from settings import (
    HTTPX_REPO_DIR,
    ROUTER_MODEL,
    SYNTHESIS_MODEL,
)

from langgraph.graph import END, StateGraph

from src.agent.nodes.conversation_memory import conversation_memory_node
from src.agent.nodes.input_preprocessor import input_preprocessor_node
from src.agent.nodes.retrieval import retrieval_node
from src.agent.nodes.router import router_node
from src.agent.nodes.security_gateway import security_gateway_node
from src.agent.nodes.synthesis import synthesis_node
from src.agent.routing import (
    route_after_conversation_memory,
    route_after_input_preprocessor,
    route_after_security_gateway,
)
from src.agent.state import AgentState
from src.llm import initialize_llm_cache

__all__ = [
    "build_code_rag_graph",
    "get_compiled_graph",
]

logger = logging.getLogger(__name__)


def build_code_rag_graph() -> Any:
    """Build the Code RAG Agent graph with v1.2 architecture.

    Graph structure:
    ```
    START → security_gateway → [SECURITY CHECK]
          → END (if blocked)
          → input_preprocessor → [HISTORY CHECK]
          → conversation_memory (if history query, skip retrieval)
          → router → retrieval → synthesis → conversation_memory
          → security_gateway (loop) OR END
    ```

    Nodes (6 total):
    1. **security_gateway**: ProtectAI DeBERTa v3 prompt injection detection
       - Validates user query for malicious content
       - Blocks if threat detected (is_blocked=True)

    2. **input_preprocessor**: Query cleaning and history detection
       - Removes filler words from queries
       - Detects conversation history requests
       - Routes history queries directly to conversation_memory

    3. **router**: Query decomposition with fast path optimization
       - Fast path: Regex-based routing (<10ms) for simple queries
       - LLM router: Gemini for complex queries (~1.5s)
       - Uses codebase tree for intelligent folder/file selection

    4. **retrieval**: Hybrid search + context expansion (integrated)
       - BM25 (weight: 0.4) with code-aware tokenization
       - Vector (weight: 1.0) with FAISS lazy loading
       - Context expansion: parent class, sibling methods, child sections

    5. **synthesis**: Streaming answer generation with citations
       - Streams response to stdout for real-time feedback
       - Formats code references with file:line citations

    6. **conversation_memory**: Combined save + reset operations
       - Stores (query, answer, citations, timestamp) tuple
       - Controls conversation loop via external flag

    Returns:
        Configured StateGraph instance (not yet compiled)

    Example:
        >>> graph = build_code_rag_graph()
        >>> app = graph.compile()
        >>> result = app.invoke({"user_query": "How does BM25 tokenization work?"})
        >>> print(result["final_answer"])
    """
    logger.info("Building Code RAG Agent graph (v1.2)...")

    # Initialize LLM cache
    initialize_llm_cache()
    logger.info(f"[SUCCESS] LLM cache initialized (router: {ROUTER_MODEL}, synthesis: {SYNTHESIS_MODEL})")

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes (6 nodes total)
    graph.add_node("security_gateway", security_gateway_node)
    graph.add_node("input_preprocessor", input_preprocessor_node)
    graph.add_node("router", partial(router_node, repo_root=HTTPX_REPO_DIR))
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("conversation_memory", conversation_memory_node)

    logger.debug("[SUCCESS] Added 6 nodes")

    # Set entry point
    graph.set_entry_point("security_gateway")

    # Add edges

    # Security gateway → END if blocked, input_preprocessor if safe
    graph.add_conditional_edges(
        "security_gateway",
        route_after_security_gateway,
        {
            "input_preprocessor": "input_preprocessor",
            "END": END,
        },
    )

    # Input preprocessor → four possible paths:
    # 1. conversation_memory: History query (final_answer already set)
    # 2. synthesis: Follow-up with sufficient history (skip retrieval)
    # 3. router: New question or follow-up needing fresh retrieval
    # 4. security_gateway: Out of scope query (route back to start)
    graph.add_conditional_edges(
        "input_preprocessor",
        route_after_input_preprocessor,
        {
            "conversation_memory": "conversation_memory",  # History query shortcut
            "synthesis": "synthesis",  # Follow-up (history sufficient)
            "router": "router",  # Normal flow (needs retrieval)
            "security_gateway": "security_gateway",  # Out of scope (back to start)
        },
    )

    # Linear pipeline: router → retrieval → synthesis → conversation_memory
    graph.add_edge("router", "retrieval")
    graph.add_edge("retrieval", "synthesis")
    graph.add_edge("synthesis", "conversation_memory")

    # Conversation loop: back to security_gateway for new queries, or end
    graph.add_conditional_edges(
        "conversation_memory",
        route_after_conversation_memory,
        {
            "security_gateway": "security_gateway",  # Loop back (new query needs validation)
            "END": END,
        },
    )

    logger.debug("[SUCCESS] Added all edges")
    logger.info("[SUCCESS] Code RAG Agent graph (v1.2) built successfully")

    return graph


def get_compiled_graph(
    checkpointer: Any | None = None,
) -> Any:
    """Get a compiled Code RAG Agent graph ready for execution.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled graph application

    Example:
        >>> app = get_compiled_graph()
        >>> result = app.invoke({"user_query": "How does BM25 work?"})
        >>> print(result["final_answer"])
        >>> print(result["citations"])
    """
    graph = build_code_rag_graph()
    logger.info("Compiling graph...")
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("[SUCCESS] Graph compiled successfully")
    return compiled
