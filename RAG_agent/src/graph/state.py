"""State schema for the LangGraph documentation assistant.

This module defines the shared state structure used across all graph nodes.
"""

from typing import List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from models.intent_classification import IntentClassification
from models.conversation_memory import ConversationMemory

__all__ = ["AgentState"]


class AgentState(TypedDict, total=False):
    """State schema for the documentation assistant agent.

    This state is shared across all nodes in the LangGraph workflow.
    Each node can read from and write to this state.

    Attributes:
        # User input
        user_input: The user's original request
        cleaned_request: Cleaned/parsed version of the request
        code_snippets: Extracted code from user input
        extracted_data: Extracted structured data (errors, configs)

        # Intent classification
        intent_result: Intent classification result

        # RAG retrieval
        retrieved_documents: Top-5 documents from hybrid retrieval (as dicts for serialization)
        retrieval_metadata: Debugging/analysis metadata for retrieval

        # Agent processing
        messages: Conversation messages for the agent
        final_response: The agent's final response to the user

        # Security validation
        gateway_passed: Whether request passed security validation
        gateway_reason: Reason for security validation failure

        # Conversation memory
        conversation_memory: Complete conversation history
        continue_conversation: Flag to continue conversation or end

    Example:
        >>> state = AgentState(
        ...     user_input="How do I add persistence to LangGraph?",
        ...     cleaned_request="How do I add persistence to LangGraph?",
        ...     intent_result=IntentClassification(
        ...         framework="langgraph",
        ...         language="python",
        ...         keywords=["persistence"],
        ...         topics=["persistence"],
        ...         intent_type="implementation_guide",
        ...         requires_rag=True
        ...     )
        ... )
    """

    # User input
    user_input: str
    cleaned_request: Optional[str]
    code_snippets: Optional[List[str]]
    extracted_data: Optional[dict]

    # Intent classification
    intent_result: Optional[IntentClassification]

    # RAG retrieval
    retrieved_documents: Optional[List[dict]]
    retrieval_metadata: Optional[dict]

    # Agent processing
    messages: list[BaseMessage]
    # Note: streaming_response is not in state schema because generators can't be serialized
    # Streaming responses are consumed immediately and not persisted
    final_response: Optional[str]

    # Security validation
    gateway_passed: Optional[bool]
    gateway_reason: Optional[str]

    # Conversation memory
    conversation_memory: Optional[ConversationMemory]
    continue_conversation: Optional[bool]
