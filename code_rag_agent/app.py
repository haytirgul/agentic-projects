"""
Simple command-line chatbot for the LangGraph Documentation Assistant.

This chatbot interacts with users via the command line, handling:
- Request parsing and intent classification
- Tool-based clarification (agent uses ask_user tool when needed)
- RAG-based responses (placeholder for now)

The agent decides dynamically when to ask for clarification using the
ask_user tool, providing a more natural conversational flow.

Note: Gateway validation has been removed to reduce overhead.
We assume users are using the agent properly.

Usage:
    python chatbot.py

Commands:
    - Type your question to interact with the agent
    - Type 'quit' or 'exit' to end the session
    - Type 'help' for usage instructions
"""

import sys
import time
import uuid
import logging
import warnings
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from settings import LANGCHAIN_TRACING_V2

# Suppress all warnings (LangSmith UUID warnings, Pydantic warnings, etc.)
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  LangChain/LangGraph/LangSmith Documentation Assistant")
    print("=" * 70)
    print("\nWelcome! I can help you with:")
    print("  - LangChain: agents, RAG, chains, tools, memory")
    print("  - LangGraph: StateGraph, persistence, multi-agent systems")
    print("  - LangSmith: tracing, evaluation, deployment")
    print("\nType 'quit' or 'exit' to end the session.")
    print("Type 'help' for more information.")
    print("=" * 70 + "\n")


def print_help() -> None:
    """Print help information."""
    print("\n" + "-" * 70)
    print("HELP - How to use this chatbot:")
    print("-" * 70)
    print("1. Ask any question about LangChain, LangGraph, or LangSmith")
    print("2. If your question is vague, I may ask for clarification")
    print("3. Provide clarification to get a more specific answer")
    print("\nExamples:")
    print("  - How do I create a LangChain agent?")
    print("  - How do I add persistence to LangGraph?")
    print("  - What's the difference between StateGraph and MessageGraph?")
    print("\nCommands:")
    print("  - 'quit' or 'exit': End the session")
    print("  - 'help': Show this help message")
    print("  - 'clear': Start a new conversation")
    print("-" * 70 + "\n")


def process_question(
    app: Any,
    question: str,
    thread_id: str,
    conversation_memory: Any = None,
) -> Any:
    """Process a user question through the graph.

    The graph will:
    1. Parse and clean the request
    2. Classify the intent
    3. Run hybrid RAG retrieval
    4. Run the agent (which may use ask_user tool for clarification)
    5. Save the turn to conversation memory
    6. Return the final response and updated memory

    Tool-based clarification happens automatically during agent execution -
    the ask_user tool prompts the user directly when the agent decides
    clarification would be helpful.

    Args:
        app: Compiled graph application
        question: User's question
        thread_id: Session thread ID for checkpointer
        conversation_memory: Optional conversation memory from previous turn

    Returns:
        Updated conversation memory from this turn
    """
    # TIMING: Start tracking this query
    from src.utils.timing_logger import start_query_timing, end_query_timing, print_timing_report
    query_id = f"q_{thread_id}_{int(time.time() * 1000)}"
    start_query_timing(query_id, question)

    config = {"configurable": {"thread_id": thread_id}}

    print("\n[>] Processing your question...")

    try:
        # Prepare state with question and memory
        initial_state = {"user_input": question}
        if conversation_memory:
            initial_state["conversation_memory"] = conversation_memory

        # Set continue_conversation=False to stop after one turn
        initial_state["continue_conversation"] = False

        # Run the graph - it will execute one full turn then stop at prompt_continue
        result = app.invoke(initial_state, config)

        # TIMING: End tracking and print report
        end_query_timing(query_id)
        # print_timing_report(query_id)

        # Return the updated conversation memory for next turn
        return result.get("conversation_memory")

    except Exception as e:
        end_query_timing(query_id)
        print(f"\n[X] Error processing question: {e}")
        raise


def run_chatbot() -> None:
    """Main chatbot loop with conversation memory."""
    print_banner()

    # Setup memory
    memory = MemorySaver()
    app = None  # Will be initialized lazily or eagerly based on settings


    # Eager initialization (traditional behavior)
    print("[*] Initializing agent (loading LLM and RAG components in parallel)...")
    from src.graph import get_compiled_graph
    app = get_compiled_graph(checkpointer=memory)
    print("[+] Agent ready with LLM logging enabled!\n")


    # Display LangSmith status
    if LANGCHAIN_TRACING_V2:
        print("[*] LangSmith tracing: ENABLED")
        print("    View traces at: https://smith.langchain.com")
        logger.info("LangSmith tracing enabled")
    else:
        print("[*] LangSmith tracing: DISABLED")
        print("    To enable: Set LANGCHAIN_TRACING_V2=true in .env")
        print("    Get free API key at: https://smith.langchain.com")

    # Create a session ID for this conversation
    session_id = str(uuid.uuid4())

    # Track conversation memory across turns
    conversation_memory = None

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                if conversation_memory and len(conversation_memory.turns) > 0:
                    print(f"\n[i] This conversation had {len(conversation_memory.turns)} turn(s).")
                print("\nGoodbye! Thanks for using the assistant.")
                break

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "clear":
                session_id = str(uuid.uuid4())
                conversation_memory = None
                print("\n[>] Started new conversation.")
                continue

            # Lazy initialization: build graph on first message
            if app is None:
                print("\n[*] First message detected - initializing agent now...")
                print("[*] Loading LLM and RAG components in parallel...")
                from src.graph import get_compiled_graph
                app = get_compiled_graph(checkpointer=memory)
                print("[+] Agent initialized and ready!\n")

            # Process the question and get updated memory
            conversation_memory = process_question(
                app,
                user_input,
                session_id,
                conversation_memory
            )

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break

        except Exception as e:
            print(f"\n[X] Error: {e}")
            print("Please try again or type 'help' for assistance.")


if __name__ == "__main__":
    try:
        print("Starting the chatbot...")
        run_chatbot()
    except Exception as e:
        print(f"\n[X] Fatal error: {e}")
        sys.exit(1)
