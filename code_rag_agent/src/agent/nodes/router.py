"""Router node for query decomposition with fast path optimization.

This node decomposes user queries into structured retrieval requests using:
- Fast path: Regex-based routing for simple queries (<10ms)
- LLM router: Gemini Flash for complex queries (~1.5s)

Author: Hay Hoffman
"""

import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from models.retrieval import RouterOutput
from prompts.router_prompt import build_router_prompt, get_cached_codebase_tree
from settings import ROUTER_MAX_TOKENS, ROUTER_MODEL, ROUTER_TEMPERATURE

from src.agent.state import AgentState
from src.llm import get_cached_llm, invoke_structured
from src.retrieval import FastPathRouter

__all__ = ["router_node"]

logger = logging.getLogger(__name__)

# Router system message
ROUTER_SYSTEM_MESSAGE = "You are a codebase navigation expert that outputs JSON."


def _route_with_llm(user_query: str, codebase_tree: str) -> RouterOutput:
    """Route query using LLM with structured output.

    Args:
        user_query: User's question
        codebase_tree: Directory tree representation

    Returns:
        RouterOutput with retrieval requests
    """
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_MESSAGE),
        HumanMessage(content=build_router_prompt(user_query, codebase_tree)),
    ]

    router_output = invoke_structured(
        messages=messages,
        output_schema=RouterOutput,
        llm=get_cached_llm(ROUTER_MODEL),
        temperature=ROUTER_TEMPERATURE,
        max_tokens=ROUTER_MAX_TOKENS,
    )

    logger.info(f"LLM router generated {len(router_output.retrieval_requests)} requests")
    return router_output


def router_node(state: AgentState, repo_root: Path) -> dict[str, Any]:
    """Route user query with fast path optimization.

    This node:
    1. Checks fast path router for simple queries (regex-based, <10ms)
    2. Falls back to LLM router for complex queries (Gemini Flash, ~1.5s)
    3. Generates structured RouterOutput with RetrievalRequests

    Args:
        state: Current agent state containing user_query
        repo_root: Repository root path for codebase tree generation

    Returns:
        Updated state with:
            - router_output: RouterOutput with retrieval requests
            - codebase_tree: Filtered directory tree for context

    Raises:
        ValueError: If user_query is missing from state

    Example:
        >>> state = {"user_query": "How does BM25 tokenization work?"}
        >>> result = router_node(state, Path("/repo"))
        >>> result["router_output"].retrieval_requests[0].query
        "BM25 tokenize split camelCase"
    """
    user_query = state.get("user_query")
    if not user_query:
        raise ValueError("user_query is required for router node")

    logger.info(f"Router processing query: {user_query[:100]}...")

    try:
        # Get codebase tree (cached in router_prompt module)
        codebase_tree = get_cached_codebase_tree(repo_root)

        # Try fast path first
        fast_result = FastPathRouter().route(user_query)

        if fast_result:
            logger.info("[FAST PATH] Router matched (<10ms)")
            router_output = fast_result
        else:
            logger.info("[LLM PATH] Using LLM router (~1.5s)")
            router_output = _route_with_llm(user_query, codebase_tree)

        logger.info(f"Router generated {len(router_output.retrieval_requests)} retrieval requests")

        return {
            "router_output": router_output,
            "codebase_tree": codebase_tree,
        }

    except Exception as e:
        logger.error(f"Router node failed: {e}", exc_info=True)
        return {"error": f"Router failed: {str(e)}"}
