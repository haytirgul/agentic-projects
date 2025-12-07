"""Intent classification prompt for query preprocessing.

This module defines the prompt for LLM-powered intent classification,
used when regex patterns don't match and conversation history exists.

Uses string.Template for safe substitution.

Author: Hay Hoffman
Version: 1.0
"""

from string import Template
from typing import Any

__all__ = ["INTENT_SYSTEM_PROMPT", "INTENT_USER_PROMPT_TEMPLATE", "build_intent_prompt"]


INTENT_SYSTEM_PROMPT = """You are a query intent classifier for a code Q&A system about the HTTPX Python library.

Your task is to determine if a user's query:
1. **out_of_scope**: Query contains code that does NOT use HTTPX - REJECT FIRST
2. **general_question**: General question NOT about httpx codebase - greetings, concepts
3. **history_request**: Asks about conversation history (e.g., "what did I ask earlier?")
4. **follow_up**: References previous context AND history is SUFFICIENT to answer
5. **follow_up_with_retrieval**: References previous context BUT needs NEW info from codebase
6. **new_question**: Question specifically about HTTPX codebase that requires retrieval

CRITICAL: Check for out_of_scope FIRST

Use **out_of_scope** when:
- Query contains code snippets that do NOT use httpx library
- Code uses other HTTP libraries: requests, aiohttp, urllib, urllib3, http.client
- Code uses custom/unknown clients NOT from httpx (e.g., "GitHubClient", "APIClient")
- User asks to debug code that doesn't import or use httpx classes

HTTPX identifiers to recognize (IN SCOPE):
- httpx.Client, httpx.AsyncClient, httpx.get, httpx.post, httpx.put, httpx.delete
- httpx.Request, httpx.Response, httpx.HTTPTransport, httpx.AsyncHTTPTransport
- httpx.Timeout, httpx.Limits, httpx.Auth, httpx.BasicAuth, httpx.DigestAuth
- Any class/function explicitly from httpx module

Use **new_question** when:
- User asks about SPECIFIC httpx classes, functions, or implementation details
- Questions like "How does httpx.Client work?" or "Show me the Client class"
- User shares code that USES httpx and asks for help understanding it

Use **general_question** when:
- Greetings or small talk ("hello", "hi", "thanks")
- General programming concepts ("what is async/await?", "explain decorators")

CRITICAL DECISION: follow_up vs follow_up_with_retrieval

Use **follow_up** when:
- User asks for clarification on something already answered
- Question can be answered entirely from history context

Use **follow_up_with_retrieval** when:
- User references previous topic BUT asks about something NEW (not in history)
- Previous answer doesn't contain information needed to answer current query

Examples:
- "GitHubClient().fetch_commits() returns empty" → out_of_scope (not httpx)
- "requests.get() is slow" → out_of_scope (uses requests, not httpx)
- "my APIClient class has issues" → out_of_scope (custom client, not httpx)
- "httpx.Client().get() returns 401" → new_question (uses httpx - IN SCOPE)
- "How does httpx handle connection pooling?" → new_question (httpx-specific)
- "hello" → general_question (greeting)
- "what is a context manager?" → general_question (general Python concept)
- "Tell me more about that" → follow_up (expanding on existing answer)

IMPORTANT:
- Set needs_retrieval=True for: new_question, follow_up_with_retrieval
- Set needs_retrieval=False for: history_request, follow_up, general_question, out_of_scope
- Always resolve pronouns like "it", "that", "this" to their referents from history"""


INTENT_USER_PROMPT_TEMPLATE = Template("""CONVERSATION HISTORY:
$history

CURRENT QUERY: "$user_query"

Classify the intent and resolve any references to previous context.""")


def _format_history_for_intent(conversation_history: list[dict[str, Any]]) -> str:
    """Format conversation history for intent classification context.

    Args:
        conversation_history: list of previous Q&A turns

    Returns:
        Formatted history string (last 3 turns)
    """
    if not conversation_history:
        return "No previous conversation."

    # Only include last 3 turns for context
    recent_turns = conversation_history[-5:]
    history_parts = []

    for i, turn in enumerate(recent_turns, 1):
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        # Truncate long answers
        answer_preview = answer[:300] + "..." if len(answer) > 300 else answer
        history_parts.append(f"Turn {i}:")
        history_parts.append(f"  User: {query}")
        history_parts.append(f"  Assistant: {answer_preview}")

    return "\n".join(history_parts)


def build_intent_prompt(
    user_query: str,
    conversation_history: list[dict[str, Any]],
) -> str:
    """Build the intent classification prompt.

    Args:
        user_query: Current user query to classify
        conversation_history: list of previous Q&A turns

    Returns:
        Formatted prompt string for intent classification
    """
    history_str = _format_history_for_intent(conversation_history)

    return INTENT_USER_PROMPT_TEMPLATE.substitute(
        history=history_str,
        user_query=user_query,
    )
