"""Command-line chatbot for the Code RAG Agent.

This chatbot interacts with users via the command line, handling:
- Security gateway validation (prompt injection detection)
- Query decomposition via router
- Hybrid RAG retrieval with context expansion
- Synthesis with streaming output

Usage:
    python app.py

Commands:
    - Type your question to interact with the agent
    - Type 'quit' or 'exit' to end the session
    - Type 'help' for usage instructions
    - Type 'clear' to start a new conversation

Author: Hay Hoffman
"""

import logging
import os
import sys
import uuid
import warnings
from typing import Any

# Suppress transformers and torch warnings BEFORE importing them
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langgraph.checkpoint.memory import MemorySaver

from settings import LANGCHAIN_TRACING_V2

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  Code RAG Agent - HTTPX Codebase Assistant")
    print("=" * 70)
    print("\nWelcome! I can help you understand the HTTPX codebase:")
    print("  - Code implementation details")
    print("  - Architecture and design patterns")
    print("  - API usage and examples")
    print("\nType 'quit' or 'exit' to end the session.")
    print("Type 'help' for more information.")
    print("=" * 70 + "\n")


def print_help() -> None:
    """Print help information."""
    print("\n" + "-" * 70)
    print("HELP - How to use this chatbot:")
    print("-" * 70)
    print("1. Ask any question about the HTTPX codebase")
    print("2. Follow-up questions are supported")
    print("3. Answers include file:line citations")
    print("\nExamples:")
    print("  - How does BM25 tokenization work?")
    print("  - What is the Client class responsible for?")
    print("  - How does connection pooling work?")
    print("\nCommands:")
    print("  - 'quit' or 'exit': End the session")
    print("  - 'help': Show this help message")
    print("  - 'clear': Start a new conversation")
    print("-" * 70 + "\n")


def process_question(
    app: Any,
    question: str,
    thread_id: str,
) -> bool:
    """Process a user question through the graph.

    v1.3: Conversation history is now managed automatically by LangGraph's
    add_messages annotation and the MemorySaver checkpointer. No need to
    pass or track conversation_history manually.

    The graph will:
    1. Validate via security gateway
    2. Preprocess input (clean query, detect follow-ups)
    3. Route query (fast path or LLM router)
    4. Retrieve chunks (hybrid BM25 + vector with context expansion)
    5. Synthesize answer with streaming output
    6. Save messages to conversation memory (via add_messages)

    Args:
        app: Compiled graph application
        question: User's question
        thread_id: Session thread ID for checkpointer (persists messages)

    Returns:
        True if successful, False if error/blocked
    """
    config = {"configurable": {"thread_id": thread_id}}

    print("\n[>] Processing your question...")

    try:
        # Prepare state with question only
        # Messages (history) are automatically loaded from checkpointer via thread_id
        initial_state: dict[str, Any] = {"user_query": question}

        # Run the graph
        result = app.invoke(initial_state, config)

        # Check for errors
        if result.get("error"):
            print(f"\n[X] Error: {result['error']}")
            return False

        # Check if blocked by security
        if result.get("is_blocked"):
            print(f"\n[!] Request blocked: {result.get('gateway_reason', 'Security violation')}")
            return False

        return True

    except Exception as e:
        print(f"\n[X] Error processing question: {e}")
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise


def run_chatbot() -> None:
    """Main chatbot loop with LangGraph-managed conversation history."""
    print_banner()

    # Setup memory checkpoint - this persists messages across invocations
    memory = MemorySaver()

    # Initialize agent
    print("[*] Initializing agent...")
    print("    - Loading graph...")
    from src.agent.graph import get_compiled_graph
    app = get_compiled_graph(checkpointer=memory)

    print("    - Pre-loading retrieval components (chunks, BM25, embeddings)...")
    from src.agent.nodes.retrieval import initialize_retrieval
    initialize_retrieval()

    print("[+] Agent ready!\n")

    # Display LangSmith status
    if LANGCHAIN_TRACING_V2:
        print("[*] LangSmith tracing: ENABLED")
        print("    View traces at: https://smith.langchain.com")
    else:
        print("[*] LangSmith tracing: DISABLED")
        print("    To enable: Set LANGCHAIN_TRACING_V2=true in .env")

    # Create a session ID for this conversation
    # Same thread_id = same conversation history (managed by MemorySaver)
    session_id = str(uuid.uuid4())
    turn_count = 0

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                if turn_count > 0:
                    print(f"\n[i] This conversation had {turn_count} turn(s).")
                print("\nGoodbye! Thanks for using the Code RAG Agent.")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "clear":
                # New session_id = new conversation (messages won't carry over)
                session_id = str(uuid.uuid4())
                turn_count = 0
                print("\n[>] Started new conversation.")
                continue

            # Process the question
            success = process_question(app, user_input, session_id)
            if success:
                turn_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n[X] Error: {e}")
            print("Please try again or type 'help' for assistance.")


if __name__ == "__main__":
    try:
        run_chatbot()
    except Exception as e:
        print(f"\n[X] Fatal error: {e}")
        sys.exit(1)
