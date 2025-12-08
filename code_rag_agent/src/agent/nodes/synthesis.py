"""Synthesis node for answer generation with citations.

This node generates grounded answers from expanded chunks with precise
file:line citations for every factual claim.

Author: Hay Hoffman
"""

import logging
import re
import sys
from typing import Any

from prompts.synthesis_prompt import build_general_question_messages, build_synthesis_messages
from settings import SYNTHESIS_MAX_TOKENS, SYNTHESIS_MODEL, SYNTHESIS_TEMPERATURE

from src.agent.state import AgentState
from src.llm import get_cached_llm

__all__ = ["synthesis_node"]

logger = logging.getLogger(__name__)

# ANSI color codes
GREEN = "\033[92m"
RESET = "\033[0m"

# Pattern to match citations like (`file:line`) or (file:line) or `file:line`
# Matches patterns like: (`docs\index.md:1`) or (httpx\_client.py:594)
CITATION_PATTERN = re.compile(
    r"(\(`[^`]+:\d+`\)|\([^()]+:\d+\)|`[^`]+:\d+`)"
)


def _colorize_citations(text: str) -> str:
    """Colorize citations in text with green ANSI color."""
    return CITATION_PATTERN.sub(lambda m: f"{GREEN}{m.group(1)}{RESET}", text)


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Generate natural language answer with real-time streaming output.

    v1.5: Uses native LLM streaming - outputs chunks as they arrive from the
    model, no artificial delays. Much faster perceived response time.

    This node:
    1. Formats expanded chunks for LLM context
    2. Includes conversation history for follow-up questions
    3. Streams response to stdout in real-time (native LLM streaming)
    4. Citations are colorized in green for visibility

    Args:
        state: Current agent state containing:
            - user_query: User's question
            - expanded_chunks: Retrieved chunks with context
            - messages: LangGraph messages (conversation history)

    Returns:
        Updated state with:
            - final_answer: Generated answer
            - citations: list of citation dicts

    Raises:
        ValueError: If required fields missing from state

    Example:
        >>> state = {
        ...     "user_query": "How does BM25 work?",
        ...     "expanded_chunks": [...]
        ... }
        >>> result = synthesis_node(state)
        >>> result["final_answer"]
        "BM25 tokenization splits camelCase..."
    """
    user_query = state.get("user_query")
    if not user_query:
        raise ValueError("user_query is required for synthesis node")

    expanded_chunks = state.get("expanded_chunks") or []
    is_general_question = state.get("is_general_question", False)
    conversation_messages = state.get("messages") or []

    logger.info(f"Synthesizing answer for: {user_query[:100]}...")

    try:
        # Build prompt
        if is_general_question or not expanded_chunks:
            llm_messages = build_general_question_messages(user_query, conversation_messages)
            logger.info("Using general question prompt")
        else:
            llm_messages = build_synthesis_messages(user_query, expanded_chunks, conversation_messages)
            logger.info("Using code-specific prompt with context")

        llm = get_cached_llm(SYNTHESIS_MODEL)

        # Stream response with fallback
        answer_chunks: list[str] = []
        logger.info("Streaming synthesis response...")
        sys.stdout.write("\n")

        try:
            for chunk in llm.stream(llm_messages, temperature=SYNTHESIS_TEMPERATURE, max_tokens=SYNTHESIS_MAX_TOKENS):
                content = getattr(chunk, 'content', None) or (str(chunk) if chunk else "")
                if isinstance(content, str):
                    answer_chunks.append(content)
                    sys.stdout.write(_colorize_citations(content))
                    sys.stdout.flush()
        except Exception as e:
            logger.warning(f"Streaming failed, falling back: {e}")
            try:
                response = llm.invoke(llm_messages, temperature=SYNTHESIS_TEMPERATURE, max_tokens=SYNTHESIS_MAX_TOKENS)
                content = response.content or ""
                answer_chunks.append(content)
                sys.stdout.write(_colorize_citations(content))
                sys.stdout.flush()
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                raise e

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Combine chunks and extract citations
        final_answer = "".join(answer_chunks)
        citations = []

        for match in CITATION_PATTERN.finditer(final_answer):
            citation_text = match.group(1)
            clean_citation = citation_text.strip("`").strip("()")
            if ":" in clean_citation:
                try:
                    file_path, line_str = clean_citation.rsplit(":", 1)
                    citations.append({
                        "file": file_path,
                        "line": int(line_str),
                        "text": citation_text,
                    })
                except ValueError:
                    continue

        logger.info(f"[SUCCESS] Synthesis complete: {len(final_answer)} chars")
        return {"final_answer": final_answer, "citations": citations}

    except Exception as e:
        logger.error(f"Synthesis node failed: {e}", exc_info=True)
        return {
            "error": f"Synthesis failed: {str(e)}",
            "final_answer": "I encountered an error generating the answer. Please try again.",
            "citations": [],
        }
