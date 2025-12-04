"""Conversation memory node for storing and managing conversation history.

This node saves each query-response cycle and prompts the user to continue.
"""

from typing import Any, Dict
import logging

from src.graph.state import AgentState
from models.conversation_memory import ConversationMemory, ConversationTurn
from models.rag_models import RetrievedDocument

__all__ = ["save_conversation_turn_node", "prompt_continue_node"]

logger = logging.getLogger(__name__)


def save_conversation_turn_node(state: AgentState) -> Dict[str, Any]:
    """
    Save the current conversation turn to memory.

    This node:
    1. Extracts query data (user input, intent, cleaned request)
    2. Extracts response data (final response, retrieved docs)
    3. Creates a ConversationTurn
    4. Adds it to ConversationMemory
    5. Returns updated memory

    Input state:
        - user_input: str
        - cleaned_request: str
        - intent_result: IntentClassification
        - code_snippets: List[str]
        - final_response: str
        - retrieved_documents: List[RetrievedDocument]
        - conversation_memory: ConversationMemory (optional)

    Output state:
        - conversation_memory: ConversationMemory (updated)

    Example:
        >>> state = {
        ...     "user_input": "How to use checkpointers?",
        ...     "cleaned_request": "How to use checkpointers in LangGraph?",
        ...     "intent_result": IntentClassification(...),
        ...     "final_response": "To use checkpointers...",
        ...     "retrieved_documents": [RetrievedDocument(...)]
        ... }
        >>> result = save_conversation_turn_node(state)
        >>> len(result["conversation_memory"].turns)
        1
    """
    # Get or initialize conversation memory
    conversation_memory = state.get("conversation_memory")
    if conversation_memory is None:
        conversation_memory = ConversationMemory()
        logger.info("Initialized new conversation memory")

    # Extract data for this turn
    user_query = state.get("user_input", "")
    cleaned_request = state.get("cleaned_request", user_query)
    intent_result = state.get("intent_result")
    code_snippets = state.get("code_snippets", [])
    final_response = state.get("final_response", "")

    # Convert retrieved_docs from dicts back to RetrievedDocument objects
    retrieved_docs_dicts = state.get("retrieved_documents", [])
    retrieved_docs = [RetrievedDocument(**doc_dict) for doc_dict in retrieved_docs_dicts]

    # Extract document paths
    doc_paths = [doc.file_path for doc in retrieved_docs] if retrieved_docs else []

    # Create conversation turn
    turn = ConversationTurn(
        user_query=user_query,
        cleaned_request=cleaned_request,
        intent_classification=intent_result,
        code_snippets=code_snippets,
        assistant_response=final_response,
        retrieved_doc_paths=doc_paths
    )

    # Add to memory
    conversation_memory.add_turn(turn)

    logger.info(
        f"Saved conversation turn #{len(conversation_memory.turns)}: "
        f"query='{user_query[:50]}...', response_length={len(final_response)}"
    )

    return {"conversation_memory": conversation_memory}


def prompt_continue_node(state: AgentState) -> Dict[str, Any]:
    """
    Prompt the user to continue the conversation.

    This node:
    1. Checks if externally controlled (chatbot mode)
    2. If external control: respects the flag and stops
    3. If autonomous mode: displays response and continues

    Supports two modes:
    - **Chatbot mode**: External control (continue_conversation=False)
    - **Autonomous mode**: Internal loop (continue_conversation=True)

    Input state:
        - final_response: str
        - conversation_memory: ConversationMemory
        - continue_conversation: Optional[bool] (for external control)

    Output state:
        - continue_conversation: bool

    Example (Chatbot mode):
        >>> state = {
        ...     "final_response": "Here's how...",
        ...     "continue_conversation": False  # External control
        ... }
        >>> result = prompt_continue_node(state)
        >>> result["continue_conversation"]
        False

    Example (Autonomous mode):
        >>> state = {
        ...     "final_response": "Here's how...",
        ...     # No continue_conversation set
        ... }
        >>> result = prompt_continue_node(state)
        >>> result["continue_conversation"]
        True
    """
    conversation_memory = state.get("conversation_memory")
    final_response = state.get("final_response", "")

    # Check if externally controlled (chatbot mode)
    # If continue_conversation is explicitly set to False, respect it
    external_control = state.get("continue_conversation") is False

    if external_control:
        logger.info("External continuation control detected (chatbot mode)")
        logger.info("Stopping after this turn - chatbot will handle next turn")
        return {"continue_conversation": False}

    # Autonomous mode: display response and continue automatically
    logger.info("=" * 80)
    logger.info("ASSISTANT RESPONSE:")
    logger.info(final_response)
    logger.info("=" * 80)

    # Show conversation context if available
    if conversation_memory and len(conversation_memory.turns) > 1:
        logger.info("\n" + conversation_memory.get_context_summary(n=2))

    # In production, this could use ask_user tool to prompt the user:
    # ```
    # response = ask_user("Do you have another question? (yes/no)")
    # continue_conversation = response.lower() in ["yes", "y", "continue"]
    # ```

    logger.info("\nReady for next question...")

    return {
        "continue_conversation": True,  # Continue in autonomous mode
    }
