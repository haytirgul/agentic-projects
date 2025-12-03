"""Agent response nodes for message preparation and final response extraction.

This module contains nodes that handle:
- prepare_agent_messages_node: Builds context-aware messages with retrieved docs
- extract_response_node: Extracts final response from agent messages
"""

from typing import Any, Dict
import logging
from string import Template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.state import AgentState
from prompts.agent_response import AGENT_SYSTEM_PROMPT
from models.rag_models import RetrievedDocument

__all__ = [
    "prepare_agent_messages_node",
    "extract_response_node",
]

logger = logging.getLogger(__name__)


def _format_document_content(doc: Any) -> str:
    """
    Extract content from a RetrievedDocument in a simple, unified way.

    Handles three content formats:
    1. Granular (content_block/section): Use matched_text or content_block
    2. Document with matched_text: Use matched_text directly
    3. Document with sections: Combine all section content

    Args:
        doc: RetrievedDocument with content in various formats

    Returns:
        Formatted content string ready for display
    """
    # Priority 1: matched_text (works for all granularities and web results)
    if hasattr(doc, 'matched_text') and doc.matched_text:
        # Add section title if available (for granular results)
        if hasattr(doc, 'section_title') and doc.section_title:
            return f"#### {doc.section_title}\n\n{doc.matched_text}"
        return doc.matched_text

    # Priority 2: content_block dict (granular retrieval fallback)
    if hasattr(doc, 'content_block') and isinstance(doc.content_block, dict):
        block_content = doc.content_block.get('content', '')
        if block_content:
            block_title = doc.content_block.get('title', '')
            if block_title:
                return f"#### {block_title}\n\n{block_content}"
            return block_content

    # Priority 3: sections list (legacy document-level format)
    if hasattr(doc, 'sections') and doc.sections:
        section_parts = []
        for section in doc.sections:
            section_parts.append(f"#### {section.title}\n\n{section.content}")
        return "\n\n".join(section_parts)

    # No content available
    return "_[No content available]_"


def prepare_agent_messages_node(state: AgentState) -> Dict[str, Any]:
    """
    Prepare messages for response agent with retrieved documents and conversation history.

    This node:
    1. Takes intent classification + retrieved documents + conversation history
    2. Builds system prompt with intent context
    3. Injects conversation history for context-aware responses
    4. Injects retrieved docs into user message
    5. Returns messages ready for response generation

    Input state:
        - intent_result: IntentClassification
        - cleaned_request: str
        - code_snippets: List[str] (optional)
        - retrieved_documents: List[RetrievedDocument] (optional)
        - conversation_memory: ConversationMemory (optional)

    Output state:
        - messages: List[BaseMessage]

    Example:
        >>> state = {
        ...     "cleaned_request": "How to use checkpointers?",
        ...     "intent_result": IntentClassification(...),
        ...     "retrieved_documents": [RetrievedDocument(...), ...]
        ... }
        >>> result = prepare_agent_messages_node(state)
        >>> len(result["messages"])
        2  # SystemMessage + HumanMessage
    """
    intent_result = state.get("intent_result")
    cleaned_request = state.get("cleaned_request") or state.get("user_input", "")
    code_snippets = state.get("code_snippets", [])

    # Convert retrieved_docs from dicts back to RetrievedDocument objects
    retrieved_docs_dicts = state.get("retrieved_documents", [])
    retrieved_docs = [RetrievedDocument(**doc_dict) for doc_dict in retrieved_docs_dicts]

    conversation_memory = state.get("conversation_memory")

    # Build system prompt
    if intent_result:
        system_prompt = Template(AGENT_SYSTEM_PROMPT).safe_substitute(
            intent_type=intent_result.intent_type,
            framework=intent_result.framework or "general",
            topics=", ".join(intent_result.topics) if intent_result.topics else "none identified",
            keywords=", ".join(map(str, intent_result.keywords[:5])) if intent_result.keywords else "none",
        )
    else:
        system_prompt = Template(AGENT_SYSTEM_PROMPT).safe_substitute(
            intent_type="unknown",
            framework="general",
            topics="none identified",
            keywords="none",
        )

    # Add conversation context information to system prompt
    if intent_result and hasattr(intent_result, 'conversation_context'):
        context_guidance = {
            "new_topic": "This is a NEW topic. Focus on fresh information from retrieved documents.",
            "continuing_topic": "This is CONTINUING the previous topic. Build upon previous context and add new details.",
            "clarification": "This is a CLARIFICATION request. Use conversation history to explain the previous answer better. NO new RAG retrieval.",
            "follow_up": "This is a FOLLOW-UP question. Acknowledge previous context and provide additional information from retrieved documents."
        }
        guidance = context_guidance.get(intent_result.conversation_context, "")
        if guidance:
            system_prompt += f"\n\n**Conversation Context**: {guidance}"

    # Add conversation history if available and relevant
    if conversation_memory and hasattr(conversation_memory, 'turns') and conversation_memory.turns:
        # For clarifications and follow-ups, include recent conversation
        if intent_result and hasattr(intent_result, 'conversation_context'):
            if intent_result.conversation_context in ["clarification", "continuing_topic", "follow_up"]:
                history_summary = conversation_memory.get_context_summary(n=2)
                system_prompt += f"\n\n{history_summary}"
                logger.info(f"Including conversation history ({len(conversation_memory.turns)} turns) for {intent_result.conversation_context}")

    # Add retrieval context if we have docs
    if retrieved_docs:
        retrieval_context = f"""## Retrieved Documentation/n/nYou have access to {len(retrieved_docs or [])} relevant document(s). Use these to answer the user's question./n/n**IMPORTANT**: Always cite sources using the file path when referencing documentation./n"""
        system_prompt += retrieval_context

    # Build user message
    user_content_parts = [cleaned_request]

    if code_snippets:
        user_content_parts.append(
            "\n\nCode context:\n" +
            "\n".join(f"```\n{c}\n```" for c in code_snippets[:2])
        )

    # Add retrieved document content
    if retrieved_docs:
        user_content_parts.append("\n\n## Documentation Context\n")

        for i, doc in enumerate(retrieved_docs, 1):
            user_content_parts.append(f"\n### Document {i}: {doc.title}")

            # Source information (web vs offline)
            is_web = doc.retrieval_metadata.get('source') == 'web'
            web_url = doc.retrieval_metadata.get('url')
            if is_web and web_url:
                user_content_parts.append(f"**Source**: [{doc.title}]({web_url}) (web)")
                user_content_parts.append(f"**URL**: {web_url}")
            else:
                user_content_parts.append(f"**Source**: `{doc.file_path}` (offline)")

            user_content_parts.append(f"**Relevance**: {doc.relevance_score:.1f}/100")
            user_content_parts.append("")  # Empty line

            # Extract and format content using helper function
            content = _format_document_content(doc)
            user_content_parts.append(content)
            user_content_parts.append("\n---")
    

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(user_content_parts)),
    ]

    logger.info(f"Prepared agent messages with {len(retrieved_docs or [])} retrieved docs")
    return {"messages": messages}


def extract_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Extract the final response from the agent's messages.

    This node finds the last non-tool AI message and extracts its content
    as the final response to the user.

    Note: Streaming responses are now consumed immediately in tool_agent_node
    and collected into AIMessages for checkpoint compatibility.

    Input state:
        - messages: List[BaseMessage]

    Output state:
        - final_response: str

    Args:
        state: Current graph state with agent messages

    Returns:
        State update with final_response

    Example:
        >>> state = {
        ...     "messages": [
        ...         SystemMessage(content="..."),
        ...         HumanMessage(content="..."),
        ...         AIMessage(content="Here's the answer"),
        ...     ]
        ... }
        >>> result = extract_response_node(state)
        >>> result["final_response"]
        "Here's the answer"
    """
    # Get messages from state
    messages = state.get("messages", [])

    # Find the last AI message without tool calls (the final response)
    final_response = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", []) or []
            if not tool_calls:
                content = msg.content
                # Handle cases where content might be a list (LangChain can return complex content)
                if isinstance(content, list):
                    # Extract text from content parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and 'text' in part:
                            text_parts.append(part['text'])
                        elif isinstance(part, dict) and 'type' in part and part['type'] == 'text':
                            text_parts.append(part.get('text', ''))
                    final_response = ' '.join(text_parts)
                else:
                    final_response = str(content) if content is not None else None
                break

    if final_response:
        logger.info(f"Extracted final response: {final_response[:100]}...")
    else:
        logger.warning("No final response found in agent messages")
        final_response = "I apologize, but I was unable to generate a response. Please try again."

    return {"final_response": final_response}
