"""
Graph builder for the LangGraph Documentation Assistant.

This module constructs the agent graph with security gateway, hybrid RAG retrieval, and conversation memory.

Graph flow:
    START -> security_gateway -> [SECURITY CHECK]
          -> END (if malicious content detected)
          -> preprocessing -> [SECURITY ANALYSIS]
               -> END (if suspicious patterns detected)
               -> intent_classification_agent -> intent_classification_tools (loop)
          -> intent_classification_finalize -> [conditional routing]
               -> hybrid_retrieval (if requires_rag=True) -> prepare_messages
               -> prepare_messages (if clarification, skip RAG)
          -> agent <-> tools (loop) -> finalize -> save_turn -> prompt_continue
          -> reset_for_next_turn -> preprocessing (conversation loop) OR END

Key design decisions:
- Security gateway: ML-based prompt injection detection using ProtectAI DeBERTa v3 (first node)
- Tool-enabled intent classification: Classification can use ask_user for clarification during analysis
- Conversation context detection: Detects new_topic, continuing_topic, clarification, follow_up
- Conditional RAG: Clarifications skip RAG retrieval and use conversation history instead
- Hybrid RAG: BM25 + embeddings + fuzzy matching for document retrieval
- Conversation memory: Stores query-response pairs with intent/retrieval metadata
- History-aware responses: Agent receives conversation history for context-aware answers
- Continuous conversation: After each response, loops back with preserved history
- Clean separation: Intent classification handles clarification, agent handles response generation
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

# Only import lightweight dependencies at module level
from settings import LEVEL_TO_MODEL

# Defer heavy imports (LangChain, nodes, etc.) to function level

__all__ = [
    "build_agent_graph",
    "get_compiled_graph",
]

logger = logging.getLogger(__name__)

# Model configuration constants
AGENT_MODEL_LEVEL = "intermediate"  # Use slow model for high-quality responses
AGENT_TEMPERATURE = 0.3
AGENT_MAX_TOKENS = 8192


def build_agent_graph():
    """
    Build the documentation assistant graph with security gateway and hybrid RAG retrieval.

    Graph structure:
    ```
    START -> security_gateway -> [SECURITY CHECK]
          -> END (if malicious content blocked)
          -> intent_classification (includes preprocessing) -> [conditional routing]
               -> hybrid_retrieval (if requires_rag=True) -> prepare_messages
               -> prepare_messages (if clarification, skip RAG)
          -> agent -> finalize -> user_output -> save_turn -> prompt_continue
          -> reset_for_next_turn -> intent_classification (conversation loop) OR END
    ```

    Flow:
    1. security_gateway: ML-based prompt injection detection
       - If SAFE: continue to intent_classification
       - If MALICIOUS: immediately END conversation (security violation)
    2. intent_classification: Parse user input + classify intent (merged node)
       - Parses and cleans request, extracts code snippets
       - Performs security analysis
       - Classifies intent with structured output
    3. [Conditional routing]: Check if RAG is needed
       - If clarification → skip to prepare_messages (use history)
       - Otherwise → proceed to hybrid_retrieval
    5. hybrid_retrieval: BM25 + embeddings + fuzzy matching → top-5 docs
    6. prepare_messages: Build context-aware messages with conversation history and retrieved docs
    7. agent: Generate response (no tools)
    8. finalize: Extract final response from messages
    9. user_output: Display response to user with optional streaming
    10. save_turn: Save query-response pair to conversation memory
    11. prompt_continue: Ask user if they want to continue conversation
    12. reset_for_next_turn: Clear per-query state (preserve history), loop back to preprocessing OR END

    Returns:
        Configured StateGraph instance (not yet compiled)

    Example:
        >>> graph = build_agent_graph()
        >>> app = graph.compile()
        >>> result = app.invoke({"user_input": "How do I add persistence to LangGraph?"})
        >>> print(result["final_response"])
    """
    # Import heavy dependencies only when building graph
    from langgraph.graph import END, StateGraph

    from src.nodes.security_gateway import security_gateway_node
    from src.nodes.intent_classification import intent_classification_node
    from src.nodes.hybrid_retrieval_vector import hybrid_retrieval_node
    from src.llm.llm_nodes import (
        simple_agent_node,
    )
    from src.llm import get_cached_llm
    from .state import AgentState
    from .routing import (
        route_after_security_gateway,
        route_after_intent_classification,
        route_after_prompt_continue,
    )
    from src.nodes.agent_response import prepare_agent_messages_node, extract_response_node
    from src.nodes.conversation_memory import save_conversation_turn_node, prompt_continue_node
    from src.nodes.state_reset import reset_for_next_turn_node
    from src.nodes.user_output import user_output_node
    from .initialization import initialize_system

    logger.info("Building agent graph with hybrid RAG...")

    # Initialize LLM cache and RAG components (parallel initialization of PKL files + VectorDB)
    # This is idempotent - safe to call multiple times
    initialize_system()

    # Get cached agent LLM
    agent_model_name = LEVEL_TO_MODEL[AGENT_MODEL_LEVEL]
    agent_llm = get_cached_llm(agent_model_name)
    logger.info(f"Agent will use {agent_model_name} ({AGENT_MODEL_LEVEL}) with temperature={AGENT_TEMPERATURE}")
    logger.info("Agent will run without tools (ask_user removed)")

    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes (security gateway is first)
    graph.add_node("security_gateway", security_gateway_node)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
    graph.add_node("prepare_messages", prepare_agent_messages_node)
    graph.add_node("agent", partial(simple_agent_node, llm=agent_llm, temperature=AGENT_TEMPERATURE, max_tokens=AGENT_MAX_TOKENS))
    graph.add_node("finalize", extract_response_node)
    graph.add_node("user_output", user_output_node)

    # Conversation memory nodes
    graph.add_node("save_turn", save_conversation_turn_node)
    graph.add_node("prompt_continue", prompt_continue_node)
    graph.add_node("reset_for_next_turn", reset_for_next_turn_node)

    logger.debug("Added all nodes (including security gateway and conversation memory)")

    # Set entry point (security gateway is now the first node)
    graph.set_entry_point("security_gateway")

    # Add conditional edges (security gateway → END if blocked, intent_classification if safe)
    graph.add_conditional_edges(
        "security_gateway",
        route_after_security_gateway,
        {
            "intent_classification": "intent_classification",
            "END": END,
        },
    )

    # Intent classification routes to RAG or prepare_messages
    # (now includes preprocessing + security analysis internally)
    graph.add_conditional_edges(
        "intent_classification",
        route_after_intent_classification,
        {
            "hybrid_retrieval": "hybrid_retrieval",
            "prepare_messages": "prepare_messages",
        },
    )

    # Hybrid retrieval pipeline
    graph.add_edge("hybrid_retrieval", "prepare_messages")
    graph.add_edge("prepare_messages", "agent")

    # Agent goes directly to finalize (no tools)
    graph.add_edge("agent", "finalize")

    # Conversation memory flow (with user output)
    graph.add_edge("finalize", "user_output")
    graph.add_edge("user_output", "save_turn")
    graph.add_edge("save_turn", "prompt_continue")

    # Conversation loop or end
    graph.add_conditional_edges(
        "prompt_continue",
        route_after_prompt_continue,
        {
            "reset_for_next_turn": "reset_for_next_turn",
            "END": END,
        },
    )

    # Loop back to intent_classification for next query
    graph.add_edge("reset_for_next_turn", "intent_classification")

    logger.debug("Added all edges (including conversation loop)")
    logger.info("Agent graph with hybrid RAG and conversation memory built successfully")

    return graph


def get_compiled_graph(
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    Get a compiled graph ready for execution.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled graph application

    Example:
        >>> app = get_compiled_graph()
        >>> result = app.invoke({"user_input": "How do I create a LangChain agent?"})
        >>> print(result["final_response"])
    """
    graph = build_agent_graph()
    logger.info("Compiling graph...")
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully")
    return compiled
