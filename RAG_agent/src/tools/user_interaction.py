"""
User interaction tools for the LangGraph agent.

Provides tools for the agent to request clarification or additional
information from the user during execution.
"""

import logging
from typing import Callable, Optional

from langchain_core.tools import BaseTool, StructuredTool, tool

from src.security import SecurityValidationError, validate_user_input

__all__ = ["ask_user", "create_ask_user_tool"]

logger = logging.getLogger(__name__)


# Default input handler (blocking CLI input with security validation)
def _default_input_handler(question: str) -> str:
    """Default handler that uses CLI input with security validation."""
    print(f"\n[?] Agent asks: {question}")

    max_attempts = 3
    for attempt in range(max_attempts):
        response = input("You: ").strip()

        if not response:
            return "User did not provide clarification."

        # Validate user response for security threats
        # Uses LangChain text splitter and batch classification
        try:
            is_safe, reason = validate_user_input(
                text=response,
                raise_on_malicious=True
            )

            # If validation passed, return the response
            logger.info("User clarification validated as safe")
            return response

        except SecurityValidationError as e:
            # Security threat detected in clarification response
            logger.warning(f"Security threat in clarification response (attempt {attempt + 1}/{max_attempts}): {e}")
            print(f"\n⚠️  Security validation failed: {str(e)}")

            if attempt < max_attempts - 1:
                print(f"Please rephrase your response ({max_attempts - attempt - 1} attempts remaining).\n")
            else:
                logger.error("Max security validation attempts reached for clarification")
                return "[Security validation failed: User response blocked due to detected security threat]"

        except Exception as e:
            # Unexpected error - allow through with warning (fail-open)
            logger.warning(f"Security validation error (degraded mode): {e}")
            return response

    return "User did not provide valid clarification."


# Global input handler - can be overridden for different UIs
_input_handler: Callable[[str], str] = _default_input_handler


def set_input_handler(handler: Callable[[str], str]) -> None:
    """
    Set a custom input handler for the ask_user tool.
    
    This allows different UIs (CLI, web, etc.) to provide their own
    input mechanism.
    
    Args:
        handler: Function that takes a question string and returns user response
        
    Example:
        >>> def web_input_handler(question: str) -> str:
        ...     # Send to websocket, wait for response
        ...     return get_websocket_response(question)
        >>> set_input_handler(web_input_handler)
    """
    global _input_handler
    _input_handler = handler
    logger.info("Custom input handler set for ask_user tool")


def reset_input_handler() -> None:
    """Reset to the default CLI input handler."""
    global _input_handler
    _input_handler = _default_input_handler
    logger.info("Input handler reset to default CLI handler")


@tool
def ask_user(question: str) -> str:
    """Ask the user for clarification or additional information.

    Use this tool sparingly - only when you genuinely cannot provide any helpful response without additional information.

    **When to use (RARELY):**
    - Extremely vague requests that could mean almost anything: "Help with stuff"
    - When you have zero context and multiple frameworks could apply equally
    - Critical missing information that makes any answer useless

    **When NOT to use (MOST CASES):**
    - You can make reasonable assumptions (e.g., assume Python, infer framework from context)
    - The question is clear enough to answer with general guidance + examples
    - Users say "I don't know", "It doesn't matter", or "Just help me"
    - You've already asked for clarification once - proceed with what you have

    **Philosophy:** Help users, don't interrogate them. One unhelpful response = stop asking and help.
    
    Args:
        question: The clarification question to ask the user. Be clear and specific.
        
    Returns:
        The user's response to your question.
        
    Examples:
        - "Which framework are you working with - LangChain, LangGraph, or LangSmith?"
        - "Are you using Python or JavaScript?"
        - "Could you share the error message you're seeing?"
        - "What version of LangGraph are you using?"
    """
    logger.info(f"Agent asking user: {question[:100]}...")
    response = _input_handler(question)
    logger.info(f"User responded: {response[:100]}...")
    return response


def create_ask_user_tool(
    input_handler: Optional[Callable[[str], str]] = None,
    max_questions: int = 2,
) -> BaseTool:
    """
    Create an ask_user tool with custom configuration.
    
    This factory function allows creating tools with:
    - Custom input handlers (for different UIs)
    - Question limits (to prevent excessive clarification)
    
    Args:
        input_handler: Custom function to handle user input.
                      If None, uses the global handler.
        max_questions: Maximum questions the agent can ask (default: 2)
        
    Returns:
        Configured BaseTool instance
        
    Example:
        >>> def my_handler(q: str) -> str:
        ...     return input(f"Custom prompt: {q}\\n> ")
        >>> tool = create_ask_user_tool(input_handler=my_handler, max_questions=1)
    """
    question_count = 0
    handler = input_handler or _input_handler
    
    def ask_with_limit(question: str) -> str:
        nonlocal question_count
        
        if question_count >= max_questions:
            logger.warning(f"Max questions ({max_questions}) reached, returning limit message")
            return (
                f"[Maximum clarification questions ({max_questions}) reached. "
                "Please proceed with the information you have.]"
            )
        
        question_count += 1
        logger.info(f"Agent asking user (question {question_count}/{max_questions}): {question[:100]}...")
        response = handler(question)
        logger.info(f"User responded: {response[:100]}...")
        return response
    
    return StructuredTool.from_function(
        func=ask_with_limit,
        name="ask_user",
        description=ask_user.description,
    )

