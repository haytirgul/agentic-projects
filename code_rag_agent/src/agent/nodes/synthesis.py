"""Synthesis node for answer generation with citations.

This node generates grounded answers from expanded chunks with precise
file:line citations for every factual claim.

Author: Hay Hoffman
Version: 1.3
"""

import logging
import re
import sys
import time
from typing import Any

from settings import SYNTHESIS_MAX_TOKENS, SYNTHESIS_MODEL, SYNTHESIS_TEMPERATURE
from langchain_core.messages import AIMessage
from prompts.synthesis_prompt import build_synthesis_messages, build_general_question_messages

from src.agent.state import AgentState
from src.llm import get_cached_llm

__all__ = ["synthesis_node"]

logger = logging.getLogger(__name__)

# ANSI color codes
GREEN = "\033[92m"
RESET = "\033[0m"

# Streaming delay for smooth word-by-word UX (in seconds)
STREAM_DELAY = 0.005  

# Pattern to match citations like (`file:line`) or (file:line) or `file:line`
# Matches patterns like: (`docs\index.md:1`) or (httpx\_client.py:594)
CITATION_PATTERN = re.compile(
    r"(\(`[^`]+:\d+`\)|\([^()]+:\d+\)|`[^`]+:\d+`)"
)


def _colorize_citations(text: str) -> str:
    """Colorize citations in text with green ANSI color.

    Args:
        text: Text potentially containing citations

    Returns:
        Text with citations wrapped in green ANSI codes
    """
    def replace_with_green(match: re.Match) -> str:
        return f"{GREEN}{match.group(1)}{RESET}"

    return CITATION_PATTERN.sub(replace_with_green, text)


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """Generate natural language answer with streaming output.

    This node:
    1. Formats expanded chunks for LLM context
    2. Includes conversation history for follow-up questions
    3. Streams response to stdout for real-time user feedback
    4. Citations are included naturally in the response text

    Args:
        state: Current agent state containing:
            - user_query: User's question
            - expanded_chunks: Retrieved chunks with context
            - conversation_history: Previous Q&A pairs

    Returns:
        Updated state with:
            - final_answer: Generated answer
            - citations: list of citation dicts
            - messages: LLM conversation messages

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
    expanded_chunks: list[dict[str, Any]] = state.get("expanded_chunks") or []
    is_general_question = state.get("is_general_question", False)

    if not user_query:
        raise ValueError("user_query is required for synthesis node")

    logger.info(f"Synthesizing answer for: {user_query[:100]}...")

    try:
        # Get conversation history
        conversation_history: list[dict[str, Any]] = (
            state.get("conversation_history") or []
        )

        # Build appropriate prompt based on question type
        if is_general_question or not expanded_chunks:
            # General question - use LLM knowledge, no context needed
            if is_general_question:
                logger.info("General question detected, using LLM knowledge (no retrieval)")
            else:
                logger.warning("No expanded chunks provided, using general question prompt")
            messages = build_general_question_messages(
                user_query, conversation_history
            )
        else:
            # Code-specific question - use retrieved context
            messages = build_synthesis_messages(
                user_query, expanded_chunks, conversation_history
            )

        # Get LLM
        llm = get_cached_llm(SYNTHESIS_MODEL)

        # Stream response for real-time user feedback (char-by-char)
        answer_chunks: list[str] = []
        output_buffer: str = ""  # Buffer for citation detection
        logger.info("Streaming synthesis response...")

        def _output_char(char: str) -> None:
            """Output a single character with colorization and delay."""
            colorized_char = _colorize_citations(char)
            sys.stdout.write(colorized_char)
            sys.stdout.flush()
            time.sleep(STREAM_DELAY)

        for chunk in llm.stream(
            messages,
            temperature=SYNTHESIS_TEMPERATURE,
            max_tokens=SYNTHESIS_MAX_TOKENS,
        ):
            content = chunk.content
            if content and isinstance(content, str):
                answer_chunks.append(content)
                output_buffer += content

                # Process buffer character by character, keeping citations together
                i = 0
                while i < len(output_buffer):
                    char = output_buffer[i]

                    # Check if we have a potential citation starting here
                    citation_started = False
                    for marker in ['(`', '(', '`']:
                        if output_buffer[i:].startswith(marker):
                            citation_started = True
                            break

                    if citation_started:
                        # Look ahead to find complete citation
                        remaining = output_buffer[i:]
                        citation_match = CITATION_PATTERN.search(remaining)

                        if citation_match:
                            # We have a complete citation - output it as a colored unit
                            citation_end = citation_match.span()[1]
                            citation_text = remaining[:citation_end]
                            # Output the complete citation with colorization
                            colorized_citation = _colorize_citations(citation_text)
                            sys.stdout.write(colorized_citation)
                            sys.stdout.flush()
                            # Apply streaming delay as if each char was output individually
                            time.sleep(STREAM_DELAY * len(citation_text))
                            i += citation_end
                        else:
                            # Incomplete citation - wait for more content
                            break
                    else:
                        # Regular character - output immediately
                        _output_char(char)
                        i += 1

                # Remove processed characters from buffer
                output_buffer = output_buffer[i:]

        # Output any remaining buffer content (should be incomplete citation)
        if output_buffer:
            for char in output_buffer:
                _output_char(char)

        # Newline after streaming
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Combine streamed chunks
        final_answer = "".join(answer_chunks)

        # Extract citations from the final answer
        citations: list[dict[str, Any]] = []
        for match in CITATION_PATTERN.finditer(final_answer):
            citation_text = match.group(1)
            # Remove surrounding markers and extract file:line
            clean_citation = citation_text.strip('`').strip('()').strip('`')
            if ':' in clean_citation:
                file_path, line_str = clean_citation.rsplit(':', 1)
                try:
                    line_num = int(line_str)
                    citations.append({
                        "file": file_path,
                        "line": line_num,
                        "text": citation_text
                    })
                except ValueError:
                    # Invalid line number, skip this citation
                    continue

        logger.info(f"[SUCCESS] Synthesis complete: {len(final_answer)} chars")

        # Create AIMessage for conversation history
        response_message = AIMessage(content=final_answer)

        return {
            "final_answer": final_answer,
            "citations": citations,
            "messages": messages + [response_message],
        }

    except Exception as e:
        logger.error(f"Synthesis node failed: {e}", exc_info=True)
        return {
            "error": f"Synthesis failed: {str(e)}",
            "final_answer": "I encountered an error generating the answer. Please try again.",
            "citations": [],
        }
