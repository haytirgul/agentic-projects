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

Agent: HTTPX's Client class manages connection pooling through the HTTPTransport
layer. Here's how it works:

1. Connection Pool Creation: When you create a Client, it initializes an
   HTTPTransport with configurable limits (httpx/_client.py:127):

       transport = HTTPTransport(
           limits=Limits(max_keepalive_connections=20, max_connections=100)
       )

2. Connection Reuse: The transport maintains a pool of keep-alive connections
   that are reused across requests (httpx/_transports/default.py:89)...

[Citations: httpx/_client.py:127, httpx/_transports/default.py:89]
```

### Example 2: Debugging HTTPX Code

```
You: I'm getting a 401 error with this code:
     client = httpx.Client()
     response = client.get("https://api.example.com/data",
                          auth=("user", "wrong_password"))
     What's happening?

Agent: The 401 Unauthorized error indicates authentication failure. Looking at
HTTPX's authentication flow:

1. BasicAuth Encoding: HTTPX encodes credentials using httpx.BasicAuth
   (httpx/_auth.py:45):

       class BasicAuth:
           def __init__(self, username, password):
               self._auth_header = self._build_auth_header(username, password)

2. Header Injection: The auth header is added to the request
   (httpx/_client.py:892)...

Troubleshooting Steps:
- Verify your credentials are correct
- Check if the API uses a different auth scheme (Bearer, Digest, etc.)
- Inspect the response headers for WWW-Authenticate hints
```

### Example 3: Follow-up Questions

```
You: How does AsyncClient differ from Client?

Agent: The key differences between AsyncClient and Client are:

1. Async/Await Pattern: AsyncClient uses async methods (httpx/_client.py:1156):

       async def get(self, url, **kwargs):
           return await self.request("GET", url, **kwargs)

2. Transport Layer: Uses AsyncHTTPTransport with asyncio
   (httpx/_transports/asgi.py:23)...

You: Show me how it handles timeouts

Agent: Building on AsyncClient, here's how timeouts work:

1. Timeout Configuration (from previous context - AsyncClient):

       async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
           response = await client.get(url)

2. Timeout Implementation (httpx/_config.py:78):
   - connect: Time to establish connection
   - read: Time to receive response
   - write: Time to send request
   - pool: Time to acquire connection from pool...
```

### Example 4: Code Architecture Questions

```
You: What design patterns does HTTPX use?

Agent: HTTPX employs several design patterns throughout its codebase:

1. Context Manager Pattern (httpx/_client.py:201):

       class Client:
           def __enter__(self):
               return self
           def __exit__(self, *args):
               self.close()

2. Transport Abstraction (httpx/_transports/base.py:15):
   - BaseTransport defines the interface
   - HTTPTransport, ASGITransport, WSGITransport implement it

3. Builder Pattern for requests (httpx/_models.py:312):

       request = Request("GET", url)
       request = request.with_headers({"X-Custom": "value"})

4. Middleware Chain for authentication (httpx/_auth.py:89)...
```

### Example 5: Out-of-Scope Detection

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

### Example 6: General Programming Questions

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

### Example 7: Conversation History

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
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │   History    │  │  Synthesis   │  │    Router    │
     │   Response   │  │ (No Retrieval│  │   (Query     │
     │              │  │   Needed)    │  │ Decomposition│
     └──────────────┘  └──────────────┘  └──────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │         Hybrid Retrieval        │
                               │   ┌─────────┐   ┌─────────┐     │
                               │   │  BM25   │ + │  FAISS  │     │
                               │   │ (0.4w)  │   │ (1.0w)  │     │
                               │   └─────────┘   └─────────┘     │
                               │         ↓ RRF Fusion ↓          │
                               │      Context Expansion          │
                               └─────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │          Synthesis              │
                               │   (Streaming + Citations)       │
                               └─────────────────────────────────┘
                                                │
                                                ▼
                               ┌─────────────────────────────────┐
                               │      Conversation Memory        │
                               │     (Save Turn + Loop)          │
                               └─────────────────────────────────┘
```

## Project Structure

```
code_rag_agent/
├── app.py                    # Interactive chatbot entry point
├── settings.py               # Configuration (system + user)
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
│   │   ├── hybrid_retriever.py   # BM25 + FAISS fusion
│   │   ├── faiss_store.py        # Vector database
│   │   ├── chunk_loader.py       # Load processed chunks
│   │   └── context_expander.py   # Parent/sibling expansion
│   │
│   ├── llm/
│   │   ├── llm.py            # LLM factory with caching
│   │   └── workers.py        # Structured output helpers
│   │
│   └── security/
│       ├── prompt_guard.py   # DeBERTa classifier
│       └── validator.py      # Input validation
│
├── models/                   # Pydantic data models
│   ├── chunk.py              # BaseChunk schema
│   ├── retrieval.py          # RouterOutput, RetrievedChunk
│   ├── intent.py             # Intent classification
│   └── conversation.py       # Conversation turn
│
├── prompts/                  # LLM prompt templates
│   ├── router_prompt.py
│   ├── synthesis_prompt.py
│   └── intent_prompt.py
│
├── scripts/data_pipeline/
│   ├── ingestion/
│   │   └── clone_httpx_repo.py
│   ├── processing/
│   │   ├── process_all_files.py
│   │   ├── chunk_code.py
│   │   └── process_markdown.py
│   └── indexing/
│       ├── build_all_indices.py
│       ├── build_faiss_index.py
│       └── build_bm25_index.py
│
├── data/
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
