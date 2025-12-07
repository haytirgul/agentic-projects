"""Synthesis prompt template for Code RAG Agent.

This module defines the prompts for generating grounded answers with citations.
The prompts are split into system and user components:
- System prompt: Contains persona and instructions for user-presentable answers
- User prompt: Contains context, conversation history, and the user's query

Uses string.Template for safe substitution (avoids issues with special characters).

Author: Hay Hoffman
Version: 1.3
"""

from string import Template
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

# System prompt for code-specific questions (with context)
SYSTEM_PROMPT = """You are the Code RAG Agent, an expert assistant for the HTTPX Python library codebase.

### WHO YOU ARE
You help developers understand the HTTPX library - a modern, async HTTP client for Python.
You provide insights on:
- Code implementation details and internal workings
- Architecture and design patterns used in HTTPX
- API usage and practical examples

### INSTRUCTIONS
1. Answer the question comprehensively using the provided context.
2. IF the context contains the answer:
   - Explain the relevant code components.
   - Provide code snippets from the context.
   - Cite every factual claim using the file path and line number (e.g., `src/main.py:42`).
3. IF the context is insufficient:
   - State clearly what is missing.
   - Answer based on what IS available, if partial information exists.
   - Do NOT hallucinate code or facts not present in the context.
4. Format your answer in Markdown for direct presentation to the user.
5. Provide clear, natural language explanations that a user can easily understand."""

# System prompt for general questions (no context needed)
GENERAL_QUESTION_SYSTEM_PROMPT = """You are the Code RAG Agent, an expert assistant for the HTTPX Python library codebase.

### WHO YOU ARE
You help developers understand the HTTPX library - a modern, async HTTP client for Python.
Your primary expertise is on HTTPX internals, but you can also answer general programming questions.

### INSTRUCTIONS
1. Answer the question clearly and comprehensively using your knowledge.
2. For programming concepts, provide clear explanations with examples when helpful.
3. Keep responses concise but informative.
4. Format your answer in Markdown for direct presentation to the user.
5. If the question is a greeting, respond naturally and briefly, mentioning you're here to help with HTTPX questions."""

# User prompt template using string.Template
USER_PROMPT_TEMPLATE = Template("""### CONTEXT
$context

### CONVERSATION HISTORY
$history

### USER QUESTION
$user_query""")


def _format_citations(expanded_chunks: list[dict[str, Any]]) -> str:
    """Format chunks into a context string with IDs for citation.

    Note: expanded_chunks is a list of serialized ExpandedChunk dicts.
    The correct field is 'base_chunk' (not 'chunk') per models/retrieval.py.
    """
    context_parts = []

    for i, chunk in enumerate(expanded_chunks):
        # Extract core info from base_chunk (correct field name)
        base_chunk = chunk.get("base_chunk", {})
        chunk_id = base_chunk.get("id", f"chunk_{i}")
        file_path = base_chunk.get("file_path", "unknown")
        content = base_chunk.get("content", "")
        start_line = base_chunk.get("start_line", 0)
        end_line = base_chunk.get("end_line", start_line)
        name = base_chunk.get("name", "")

        # Extract expanded context
        parent = chunk.get("parent_chunk")
        siblings = chunk.get("sibling_chunks", [])
        imports = chunk.get("import_chunks", [])
        child_sections = chunk.get("child_sections", [])

        # Build context block
        block = f"--- CITATION ID: {file_path}:{start_line} ---\n"
        block += f"File: {file_path}\n"
        block += f"Lines: {start_line}-{end_line}\n"
        if name:
            block += f"Name: {name}\n"

        if parent:
            parent_content = parent.get("content", "")
            parent_name = parent.get("name", "parent")
            block += f"Parent ({parent_name}): {parent_content[:200]}...\n"

        block += f"Content:\n{content}\n"

        if siblings:
            sibling_names = [s.get("name", "") for s in siblings if s.get("name")]
            if sibling_names:
                block += f"Related Methods: {', '.join(sibling_names)}\n"

        if child_sections:
            child_names = [c.get("name", "") for c in child_sections if c.get("name")]
            if child_names:
                block += f"Sub-sections: {', '.join(child_names)}\n"

        context_parts.append(block)

    return "\n\n".join(context_parts)


def _format_history(conversation_history: list[dict[str, Any]]) -> str:
    """Format conversation history for context."""
    if not conversation_history:
        return "No previous conversation."

    history_str = ""
    for turn in conversation_history[-3:]:
        history_str += f"User: {turn.get('query', '')}\n"
        history_str += f"Assistant: {turn.get('answer', '')}\n\n"

    return history_str


def build_system_prompt() -> str:
    """Build the system prompt for the synthesis LLM.

    Returns:
        The system prompt string containing persona and instructions.
    """
    return SYSTEM_PROMPT


def build_user_prompt(
    user_query: str,
    expanded_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt containing context, history, and query.

    Uses string.Template for safe substitution.

    Args:
        user_query: The user's question.
        expanded_chunks: list of retrieved chunks with context.
        conversation_history: list of previous Q&A pairs.

    Returns:
        The formatted user prompt string.
    """
    conversation_history = conversation_history or []

    context_str = _format_citations(expanded_chunks)
    history_str = _format_history(conversation_history)

    return USER_PROMPT_TEMPLATE.substitute(
        context=context_str,
        history=history_str,
        user_query=user_query,
    )


def build_synthesis_messages(
    user_query: str,
    expanded_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Build the list of messages for the LLM.

    Args:
        user_query: The user's question.
        expanded_chunks: list of retrieved chunks with context.
        conversation_history: list of previous Q&A pairs.

    Returns:
        list of [SystemMessage, HumanMessage] for LLM invocation.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        user_query, expanded_chunks, conversation_history
    )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


def build_general_question_messages(
    user_query: str,
    conversation_history: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Build messages for general questions (no retrieval context needed).

    Args:
        user_query: The user's question.
        conversation_history: list of previous Q&A pairs.

    Returns:
        list of [SystemMessage, HumanMessage] for LLM invocation.
    """
    conversation_history = conversation_history or []
    history_str = _format_history(conversation_history)

    user_prompt = f"### CONVERSATION HISTORY\n{history_str}\n\n### USER QUESTION\n{user_query}"

    return [
        SystemMessage(content=GENERAL_QUESTION_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
