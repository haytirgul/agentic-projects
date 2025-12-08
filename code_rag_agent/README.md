# Code RAG Agent for HTTPX

A production-ready Retrieval-Augmented Generation (RAG) agent that answers questions about the [HTTPX](https://github.com/encode/httpx) Python library codebase. Built with LangGraph for state machine orchestration, hybrid search (BM25 + FAISS), and Google Gemini models.

## Features

- **Hybrid Search**: Combines BM25 keyword search with FAISS vector similarity for accurate retrieval
- **Security Gateway**: ProtectAI DeBERTa v3 prompt injection detection
- **Intelligent Routing**: Fast regex path (<10ms) + LLM routing for complex queries
- **Context Expansion**: Automatically includes parent classes, sibling methods, and related code
- **Conversation Memory**: Tracks up to 5 turns for follow-up questions
- **Streaming Output**: Real-time answer generation with citation highlighting
- **Scope Validation**: Detects and rejects queries about non-HTTPX code

## Quick Start

### 1. Initialize the Project

First-time setup (downloads models, clones HTTPX repo, builds indices):

```bash
cd code_rag_agent
python init_project.py
```

This takes ~10-15 minutes and:
- Downloads embedding model (~90MB)
- Downloads security classifier (~738MB)
- Clones the HTTPX repository
- Processes all source files into chunks
- Builds FAISS and BM25 indices

### 2. Configure Environment

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=false
SECURITY_ENABLED=true
```

### 3. Run the Agent

```bash
python app.py
```

## Usage Examples

### Example 1: Understanding HTTPX Internals

```
You: How does httpx.Client handle connection pooling?

Agent:A `Client` instance in HTTPX utilizes HTTP connection pooling to improve efficiency when making multiple requests to the same host (`docs\advanced\clients.md:8-13`). Instead of establishing a new TCP connection for each request, the `Client` reuses the existing connection (`docs\advanced\clients.md:13-15`).

Here's why using a `Client` for connection pooling is beneficial (`docs\advanced\clients.md:16-19`):

*   **Reduced Latency:** Eliminates the need for handshaking for each request.
*   **Reduced CPU Usage:**  Fewer resources are used due to connection reuse.
*   **Reduced Network Congestion:** Less overhead due to fewer new connections.

The `Client` class itself, found in `httpx_client.py`, provides the structure for managing these connections (`httpx_client.py:594`). Similarly, `AsyncClient` provides the same functionality, but for async contexts (`httpx_client.py:1307`).

Note that for optimal connection pooling, you should avoid instantiating multiple client instances in a loop. Instead, use a single, shared client instance (`docs\async.md:53-56`).
```

### Example 2: Debugging HTTPX Code

```
You: I'm getting a 401 error with this code:
     client = httpx.Client()
     response = client.get("https://api.example.com/data",
                          auth=("user", "wrong_password"))
     What's happening?

Agent: A 401 error indicates that the request to "https://api.example.com/data" was unauthorized. This usually means the server requires authentication, and the credentials provided (username: "user", password: "wrong\_password") were rejected.

Here's a breakdown of how HTTPX handles authentication and how you might resolve this:

1.  **Authentication Methods:** HTTPX supports several authentication methods (`docs\quickstart.md:472-488`, `docs\advanced\authentication.md:87-197`):
    *   **Basic Authentication:** You're using basic authentication by passing a tuple `("user", "wrong_password")` as the `auth` argument (`docs\quickstart.md:473-476`).
    *   **Digest Authentication:**  For digest authentication, you would need to instantiate a `DigestAuth` object (`docs\quickstart.md:477-482`).
    *   **Custom Authentication:** HTTPX allows you to create custom authentication schemes by subclassing `httpx.Auth` (`docs\advanced\authentication.md:87-197`).

2.  **Custom Authentication Flow:** When using custom authentication, the `auth` argument can accept a callable or an instance of a subclass of `httpx.Auth` (`docs\advanced\authentication.md:89-92`). The `auth_flow` method within a custom `Auth` class allows you to modify the request, handle responses, and retry the request if necessary (`docs\advanced\authentication.md:93-197`). The `_send_handling_auth` method in both `Client` (`httpx_client.py:930-962`) and `AsyncClient` (`httpx_client.py:1645-1677`) uses the `auth_flow` to manage the authentication process.

3.  **Handling the 401 Error:** The immediate cause is the incorrect password. However, you might want to implement more robust error handling. HTTPX provides exception classes for handling errors (`docs\quickstart.md:495-535`):

    *   `RequestError`: A superclass for any exception that occurs while issuing an HTTP request (`docs\quickstart.md:497-499`).
    *   `HTTPStatusError`: Raised by `response.raise_for_status()` for responses that are not 2xx success codes (`docs\quickstart.md:503-505`).

    You can use these exceptions to handle the 401 error gracefully:

    ```python
    import httpx

    try:
        client = httpx.Client()
        response = client.get("https://api.example.com/data", auth=("user", "wrong_password"))
        response.raise_for_status()  # Raise HTTPStatusError for bad responses (4xx or 5xx)
    except httpx.HTTPStatusError as exc:
        print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
    except httpx.RequestError as exc:
        print(f"An error occurred while requesting {exc.request.url!r}.")
    ```

4.  **Possible Solutions:**

    *   **Correct Credentials:** The most straightforward solution is to use the correct username and password.
    *   **Alternative Authentication:** The API might support other authentication methods (e.g., API keys, OAuth). Check the API documentation and use the appropriate method with HTTPX.
    *   **Custom Authentication Flow:** If the API uses a complex authentication scheme (e.g., token refresh), you can implement a custom `Auth` class as described in the HTTPX documentation (`docs\advanced\authentication.md:87-197`).
    *   **HTTPX-Auth:** You can use the `HTTPX-Auth` package, which provides authentication classes to be used with HTTPX's authentication parameter (`docs\third_party_packages.md:15-17`).

In summary, a 401 error means authentication failed. Ensure you're using the correct credentials and the appropriate authentication method for the API you're accessing. Implement error handling to manage authentication failures gracefully.
```

### Example 3: Follow-up Questions

```
You: How does AsyncClient differ from Client?

Agent: The `AsyncClient` in HTTPX is designed for asynchronous operations, while the `Client` is for synchronous operations. Here's a breakdown of their key differences:

1.  **Asynchronous vs. Synchronous:**
    *   `AsyncClient` is used within `async` functions and relies on `async` methods (`docs\async.md:28-30`).
    *   `Client` is used in standard synchronous code.

2.  **Request Methods:**
    *   The request methods in `AsyncClient` are asynchronous and must be called with `await` (`docs\async.md:33-34`). For example: `response = await client.get(...)`.
    *   The available async request methods include `get`, `options`, `head`, `post`, `put`, `patch`, `delete`, `request`, and `send` (`docs\async.md:35-43`).

3.  **Client Lifecycle:**
    *   `AsyncClient` can be used as a context manager with `async with` for automatic resource management (`docs\async.md:47-49`).
    *   You can explicitly close an `AsyncClient` using `await client.aclose()` (`docs\async.md:57-60`).

4.  **Custom Authentication:**
    *   When implementing custom authentication for the async case, you can create an authentication scheme specific to async operations (`tests\client\test_auth.py:137-159`).
    *   The `test_async_auth` test demonstrates using an authentication implementation specific to the async case, which might involve I/O or concurrency primitives (`tests\client\test_auth.py:743-757`).

5.  **Context-Managed Transports:**
    *   `AsyncClient` supports context-managed transports, allowing you to define asynchronous entry and exit behavior for the transport layer (`tests\client\test_async_client.py:183-211`).

In essence, `AsyncClient` provides an asynchronous interface for making HTTP requests, suitable for use in `async` Python code, while `Client` provides a synchronous interface..
```

### Example 4: Complex Query (Multiple Retrievals)

```
You: How does HTTPX implement both sync and async HTTP/2 support?

Agent: HTTPX provides both synchronous and asynchronous support. Here's how it's implemented, based on the context:

1.  **Sync and Async API:**
    *   HTTPX supports both a standard thread-concurrency API and an async API (`CHANGELOG.md:825`).
    *   The library includes both `Client` and `AsyncClient` classes (`docs\http2.md:41`).

2.  **Async Client Usage:**
    *   Async codebases should use `httpx.AsyncClient()` instead of `httpx.Client()` (`CHANGELOG.md:827`).
    *   When upgrading to HTTPX 0.11, existing async users should ensure they are using a client instance to make requests instead of the top-level API (`CHANGELOG.md:826`).

3.  **Enabling HTTP/2:**
    *   HTTP/2 support is not enabled by default (`docs\http2.md:20`).
    *   To use HTTP/2, you need to install the optional `http2` dependencies: `pip install httpx[http2]` (`docs\http2.md:27`).
    *   You can then instantiate a client with HTTP/2 enabled:

        ```python
        client = httpx.AsyncClient(http2=True)
        ```
        (`docs\http2.md:31`)
    *   You can also use a context manager to ensure connections are properly closed:

        ```python
        async with httpx.AsyncClient(http2=True) as client:
        ...
        ```
        (`docs\http2.md:35-37`)

4.  **Usefulness of Async HTTP/2:**
    *   HTTP/2 support is available on both `Client` and `AsyncClient`, although it's typically more useful in async contexts if you're issuing lots of concurrent requests (`docs\http2.md:41-44`).
```

### Example 6: Out-of-Scope Detection

```
You: My GitHubClient().fetch_repos() is returning an empty list, what's wrong?

Agent: I can only help with questions about the HTTPX Python library.

Your query appears to involve code that doesn't use HTTPX (e.g., custom clients,
requests, aiohttp, or other HTTP libraries).

How I can help:
- Questions about httpx.Client, httpx.AsyncClient, and their methods
- Understanding HTTPX internals, connection pooling, timeouts
- Debugging code that uses HTTPX classes and functions

Please rephrase your question to be about HTTPX specifically.
```

### Example 7: General Programming Questions

```
You: What is a context manager in Python?

Agent: A context manager is a Python object that defines `__enter__` and `__exit__`
methods, enabling the `with` statement pattern for resource management.

Basic Structure:

    class MyContextManager:
        def __enter__(self):
            # Setup: acquire resource
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Cleanup: release resource
            pass

Common Use Cases:
- File handling: with open('file.txt') as f:
- Database connections
- HTTP clients (like HTTPX!)
- Locks and threading

Tip: Ask me about how HTTPX uses context managers for connection management!
```

### Example 8: Conversation History

```
You: What did we discuss earlier?

Agent: Here's a summary of our conversation:

Turn 1:
- Question: How does httpx.Client handle connection pooling?
- Answer: HTTPX's Client manages connection pooling through HTTPTransport...

Turn 2:
- Question: Show me the timeout configuration
- Answer: Timeouts are configured via httpx.Timeout with connect, read, write...

Turn 3:
- Question: What design patterns does HTTPX use?
- Answer: HTTPX uses Context Manager, Transport Abstraction, Builder patterns...
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Security Gateway                             │
│              (ProtectAI DeBERTa v3 - Prompt Injection)          │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Input Preprocessor                            │
│    (Intent Classification: history/follow-up/new/out-of-scope)  │
└─────────────────────────────────────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
       ▼                       ▼                       ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  Synthesis   │       │  Synthesis   │       │    Router    │
│  (history/   │       │  (general/   │       │   (Query     │
│  out_of_scope│       │  follow_up)  │       │ Decomposition│
│  responses)  │       │              │       │              │
└──────────────┘       └──────────────┘       └──────────────┘
       │                       │                       │
       │                       │                       ▼
       │                       │       ┌─────────────────────────────────┐
       │                       │       │         Hybrid Retrieval        │
       │                       │       │   ┌─────────┐   ┌─────────┐     │
       │                       │       │   │  BM25   │ + │  FAISS  │     │
       │                       │       │   │ (0.4w)  │   │ (1.0w)  │     │
       │                       │       │   └─────────┘   └─────────┘     │
       │                       │       │         ↓ RRF Fusion ↓          │
       │                       │       │      Context Expansion          │
       │                       │       └─────────────────────────────────┘
       │                       │                       │
       │                       │                       ▼
       │                       │       ┌─────────────────────────────────┐
       │                       │       │          Synthesis              │
       │                       │       │   (Streaming + Citations)       │
       │                       │       └─────────────────────────────────┘
       │                       │                       │
       └───────────────────────┴───────────────────────┘
                               │
                               ▼
               ┌─────────────────────────────────────┐
               │        Conversation Memory          │
               │        (Save Turn + Loop)           │
               └─────────────────────────────────────┘
```

## Project Structure

```
code_rag_agent/
├── app.py                    # Interactive chatbot entry point
├── settings.py               # Configuration (paths, models, thresholds)
├── init_project.py           # First-time setup automation
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine
│   │   ├── state.py          # AgentState schema
│   │   ├── routing.py        # Conditional edge logic
│   │   └── nodes/
│   │       ├── security_gateway.py
│   │       ├── input_preprocessor.py
│   │       ├── router.py
│   │       ├── retrieval.py
│   │       ├── synthesis.py
│   │       └── conversation_memory.py
│   │
│   ├── retrieval/
│   │   ├── hybrid_retriever.py   # BM25 + FAISS fusion with RRF
│   │   ├── faiss_store.py        # FAISS vector database
│   │   ├── chunk_loader.py       # Load processed chunks
│   │   ├── context_expander.py   # Parent/sibling expansion
│   │   ├── fast_path_router.py   # Regex-based fast routing
│   │   └── metadata_filter.py    # Filter by source_type, folders
│   │
│   ├── llm/
│   │   ├── llm.py            # LLM factory with caching
│   │   └── workers.py        # Structured output helpers
│   │
│   └── security/
│       ├── prompt_guard.py   # DeBERTa classifier
│       ├── validator.py      # Input validation
│       └── exceptions.py     # Security exceptions
│
├── models/                   # Pydantic data models
│   ├── chunk.py              # BaseChunk schema
│   ├── retrieval.py          # RouterOutput, RetrievedChunk
│   ├── synthesis.py          # SynthesisResponse, Citation
│   ├── intent.py             # QueryIntent classification
│   └── conversation.py       # ConversationTurn
│
├── prompts/                  # LLM prompt templates
│   ├── router_prompt.py      # Query decomposition prompt
│   ├── synthesis_prompt.py   # Answer generation prompt
│   └── intent_prompt.py      # Intent classification prompt
│
├── scripts/data_pipeline/
│   ├── ingestion/
│   │   └── clone_httpx_repo.py
│   ├── processing/
│   │   ├── process_all_files.py
│   │   ├── chunk_code.py         # AST-based code chunking
│   │   ├── process_markdown.py   # Header-aware markdown chunking
│   │   ├── process_text_files.py # Config file chunking
│   │   └── utils.py              # Token counting utilities
│   └── indexing/
│       ├── build_all_indices.py
│       ├── build_faiss_index.py
│       └── build_bm25_index.py
│
├── data/                     # (gitignored)
│   ├── httpx/                # Cloned HTTPX source
│   ├── processed/            # JSON chunk files
│   ├── index/                # FAISS indices
│   └── rag_components/       # BM25 index
│
└── documents/                # Architecture docs
    ├── AGENT_ARCHITECTURE.md
    ├── RETRIEVAL_ARCHITECTURE.md
    └── CHUNKING_ARCHITECTURE.md
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key (required) | - |
| `MODEL_FAST` | Fast model for routing | `gemini-2.0-flash` |
| `MODEL_INTERMEDIATE` | Synthesis model | `gemini-2.0-flash` |
| `MODEL_SLOW` | Complex reasoning model | `gemini-2.5-pro` |
| `SECURITY_ENABLED` | Enable prompt injection detection | `true` |
| `TOP_K_PER_REQUEST` | Chunks per retrieval request | `5` |
| `MAX_HISTORY_TURNS` | Conversation memory length | `5` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |

### Tuning Retrieval

```env
# Increase retrieval breadth
TOP_K_PER_REQUEST=10

# Adjust BM25 vs Vector weight
RRF_BM25_WEIGHT=0.3      # Reduce keyword importance
RRF_VECTOR_WEIGHT=1.5    # Increase semantic importance

# Context expansion
ENABLE_PARENT_EXPANSION=true
ENABLE_SIBLING_EXPANSION=true
MAX_SIBLINGS=3
```

## Intent Classification

The agent classifies queries into 6 intents:

| Intent | Description | Retrieval |
|--------|-------------|-----------|
| `new_question` | HTTPX-specific question | Yes |
| `follow_up` | Clarification on previous answer | No |
| `follow_up_with_retrieval` | Follow-up needing new info | Yes |
| `history_request` | "What did I ask earlier?" | No |
| `general_question` | Greetings, general Python | No |
| `out_of_scope` | Non-HTTPX code | No (rejected) |

## API Usage

### Programmatic Access

```python
from src.agent.graph import get_compiled_graph
from langgraph.checkpoint.memory import MemorySaver

# Create graph with memory
memory = MemorySaver()
app = get_compiled_graph(checkpointer=memory)

# Invoke with session tracking
result = app.invoke(
    {"user_query": "How does httpx.Client work?"},
    {"configurable": {"thread_id": "session_123"}}
)

print(result["final_answer"])
print(result["citations"])
```
## Author

Hay Hoffman
