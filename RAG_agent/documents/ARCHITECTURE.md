# RAG Documentation Assistant - Architecture Documentation

**Version:** 3.0 (Production Ready with RAG)
**Last Updated:** 2025-12-03
**Author:** Hay Hoffman
**Status:** Production Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Graph Structure](#graph-structure)
4. [System Initialization](#system-initialization)
5. [Node Specifications](#node-specifications)
6. [State Management](#state-management)
7. [Key Features](#key-features)

---

## Executive Summary

The RAG Documentation Assistant is a production-ready LangGraph agent that helps developers with LangChain, LangGraph, and LangSmith questions. The system uses a state machine architecture with:

- **Security Gateway** - ML-based prompt injection detection (ProtectAI DeBERTa v3)
- **Intent Classification** - Understands user queries and extracts metadata (includes preprocessing)
- **Hybrid RAG Retrieval** - VectorDB + BM25 for accurate document retrieval
- **Response Generation** - LLM-powered answers with citations
- **Conversation Memory** - Multi-turn conversation support with history tracking

### Key Features

✅ **LLM Caching** - All 3 model tiers pre-loaded at startup for performance
✅ **Parallel Initialization** - LLM cache + RAG components load concurrently
✅ **Security-First** - ML-based prompt injection detection (optional, can be disabled)
✅ **Hybrid RAG** - Combines vector search (ChromaDB) and BM25 keyword matching
✅ **Intent-Aware** - 10 intent types for optimized retrieval
✅ **Stateful Conversations** - Maintains context across multiple turns
✅ **Production Ready** - Comprehensive error handling, logging, and monitoring

---

## System Architecture

### High-Level Flow

```
┌─────────────┐
│   User      │
│   Input     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ SECURITY_GATEWAY │  ← ML-based prompt injection detection
│ (ProtectAI)      │
└──────┬───────────┘
       │
   [SAFE?]
       │
       ├─→ MALICIOUS → END
       │
       ▼ SAFE
┌──────────────────────┐
│ INTENT_CLASSIFICATION│  ← Includes preprocessing internally
│ (Parse + Classify)   │
└──────┬───────────────┘
       │
   [NEEDS RAG?]
       │
       ├─→ YES → ┌─────────────────┐
       │          │ HYBRID_RETRIEVAL│  ← VectorDB + BM25
       │          └────────┬────────┘
       │                   │
       └─→ NO (clarif.)    │
                │          │
                ▼          ▼
          ┌──────────────────┐
          │ PREPARE_MESSAGES │  ← Build context with history + docs
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │      AGENT       │  ← Generate response (no tools)
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    FINALIZE      │  ← Extract final response
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │   USER_OUTPUT    │  ← Display to user (streaming)
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │    SAVE_TURN     │  ← Save to conversation memory
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │ PROMPT_CONTINUE  │  ← Ask to continue?
          └────────┬─────────┘
                   │
            [CONTINUE?]
                   │
                   ├─→ YES → RESET_FOR_NEXT_TURN → INTENT_CLASSIFICATION (loop)
                   │
                   └─→ NO → END
```

### Architecture Principles

1. **Security First** - All inputs validated before processing
2. **Caching for Performance** - LLM instances cached at startup
3. **Parallel Initialization** - Components load concurrently for speed
4. **Fail-Fast Validation** - Invalid inputs rejected early
5. **Stateful Conversations** - Context maintained across turns
6. **No Fuzzy Matching** - Removed for performance (relies on vector + BM25)

---

## Graph Structure

### Node Inventory

| Node Name | Purpose | LLM Calls | Key Operations |
|-----------|---------|-----------|----------------|
| `security_gateway` | Prompt injection detection | 0 (ML model) | ProtectAI DeBERTa v3 classification |
| `intent_classification` | Parse + classify intent | 1 | Extract framework, language, topics, intent type |
| `hybrid_retrieval` | Retrieve relevant docs | 0 | VectorDB + BM25 → top-5 results |
| `prepare_messages` | Build LLM context | 0 | Inject docs + conversation history |
| `agent` | Generate response | 1 | LLM response generation (no tools) |
| `finalize` | Extract response | 0 | Extract text from AI message |
| `user_output` | Display to user | 0 | Print response (with streaming) |
| `save_turn` | Save conversation | 0 | Store query, response, intent, docs |
| `prompt_continue` | Ask to continue | 0 | Prompt user for next query |
| `reset_for_next_turn` | Reset state | 0 | Clear per-query state, keep history |

### Edge Routing Logic

```python
# Security Gateway Routing
def route_after_security_gateway(state):
    """Route based on security check result."""
    if state["security_check"]["is_safe"]:
        return "intent_classification"
    else:
        return "END"  # Block malicious input

# Intent Classification Routing
def route_after_intent_classification(state):
    """Route based on whether RAG retrieval is needed."""
    intent_result = state["intent_result"]

    # Skip RAG for clarifications (use conversation history instead)
    if intent_result.conversation_context == "clarification":
        return "prepare_messages"

    # Perform RAG for all other queries
    return "hybrid_retrieval"

# Conversation Loop Routing
def route_after_prompt_continue(state):
    """Route based on whether user wants to continue."""
    if state.get("continue_conversation", False):
        return "reset_for_next_turn"
    else:
        return "END"
```

### Graph Configuration

```python
# File: src/graph/builder.py

from langgraph.graph import StateGraph, END
from src.graph.state import AgentState

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("security_gateway", security_gateway_node)
graph.add_node("intent_classification", intent_classification_node)
graph.add_node("hybrid_retrieval", hybrid_retrieval_node)
graph.add_node("prepare_messages", prepare_agent_messages_node)
graph.add_node("agent", agent_node)
graph.add_node("finalize", extract_response_node)
graph.add_node("user_output", user_output_node)
graph.add_node("save_turn", save_conversation_turn_node)
graph.add_node("prompt_continue", prompt_continue_node)
graph.add_node("reset_for_next_turn", reset_for_next_turn_node)

# Set entry point (security gateway is first)
graph.set_entry_point("security_gateway")

# Add conditional edges
graph.add_conditional_edges(
    "security_gateway",
    route_after_security_gateway,
    {"intent_classification": "intent_classification", "END": END}
)

graph.add_conditional_edges(
    "intent_classification",
    route_after_intent_classification,
    {"hybrid_retrieval": "hybrid_retrieval", "prepare_messages": "prepare_messages"}
)

# Linear edges
graph.add_edge("hybrid_retrieval", "prepare_messages")
graph.add_edge("prepare_messages", "agent")
graph.add_edge("agent", "finalize")
graph.add_edge("finalize", "user_output")
graph.add_edge("user_output", "save_turn")
graph.add_edge("save_turn", "prompt_continue")

# Conversation loop
graph.add_conditional_edges(
    "prompt_continue",
    route_after_prompt_continue,
    {"reset_for_next_turn": "reset_for_next_turn", "END": END}
)

graph.add_edge("reset_for_next_turn", "intent_classification")
```

---

## System Initialization

### Parallel Component Loading

The system uses parallel initialization for optimal startup performance:

```python
# File: src/graph/initialization.py

def initialize_system():
    """
    Initialize all system components in parallel:
    1. LLM Cache: All 3 Gemini model instances
    2. RAG Components: PKL files (DocumentIndex, BM25) + VectorDB
    3. Security: ProtectAI DeBERTa v3 model (optional)

    This function is:
    - Thread-safe: Uses a lock to prevent concurrent initialization
    - Idempotent: Safe to call multiple times (only initializes once)
    - Parallel: All components load simultaneously for maximum speed
    """

    # Run all initializations in parallel threads
    llm_thread = threading.Thread(target=init_llm)
    rag_thread = threading.Thread(target=init_rag)
    security_thread = threading.Thread(target=init_security)

    llm_thread.start()
    rag_thread.start()
    security_thread.start()

    llm_thread.join()
    rag_thread.join()
    security_thread.join()
```

### LLM Caching

All LLM instances are pre-created and cached at startup:

```python
# File: src/llm/llm.py

# Global cache for LLM instances
_LLM_CACHE: Dict[str, BaseChatModel] = {}

def initialize_llm_cache():
    """Pre-create and cache the 3 model tier instances."""
    models_to_cache = [MODEL_FAST, MODEL_INTERMEDIATE, MODEL_SLOW]

    # Cache models in parallel
    with ThreadPoolExecutor(max_workers=len(models_to_cache)) as executor:
        futures = [executor.submit(_cache_model, model) for model in models_to_cache]
        for future in as_completed(futures):
            future.result()

def get_cached_llm(model: str) -> BaseChatModel:
    """Get a cached LLM instance (no recreation overhead)."""
    return _LLM_CACHE[model]
```

**Benefits:**
- Eliminates LLM recreation overhead on every graph invocation
- Reduces latency by ~200-500ms per query
- Memory efficient (only 3 instances total)

---

## Node Specifications

### 1. Security Gateway

**File:** `src/nodes/security_gateway.py`

**Purpose:** ML-based prompt injection detection using ProtectAI DeBERTa v3

**Input:** `user_input` (raw user query)

**Output:** `security_check` dict with `is_safe` boolean and `reason`

**Processing:**
1. Load ProtectAI DeBERTa v3 model (if SECURITY_ENABLED=true)
2. Classify input as SAFE or INJECTION
3. Block if injection detected, otherwise proceed

**Note:** Can be disabled via `SECURITY_ENABLED=false` in settings

---

### 2. Intent Classification

**File:** `src/nodes/intent_classification.py`

**Purpose:** Parse and classify user intent (includes preprocessing logic)

**Input:** `user_input` + `conversation_history`

**Output:** `intent_result` (IntentClassificationResult)

**Processing:**
1. **Preprocessing** (internal):
   - Clean input text
   - Extract code snippets
   - Detect conversation context (new_topic, continuing_topic, clarification, follow_up)
2. **Classification**:
   - Classify into 10 intent types (implementation_guide, troubleshooting, etc.)
   - Extract framework (langchain, langgraph, langsmith)
   - Extract language (python, javascript)
   - Extract topics/entities
3. **Determine if RAG is needed** (clarifications skip RAG)

**LLM Model:** Fast (gemini-2.5-flash-lite)

---

### 3. Hybrid Retrieval

**File:** `src/nodes/hybrid_retrieval_vector.py`

**Purpose:** Retrieve relevant documents using hybrid search

**Input:** `intent_result` (for query context)

**Output:** `retrieved_docs` (list of Document objects)

**Processing:**
1. **Vector Search** (ChromaDB):
   - Semantic search using sentence-transformers embeddings
   - Returns top-25 candidates
2. **BM25 Search**:
   - Keyword-based search for exact term matching
   - Returns top-25 candidates
3. **Hybrid Scoring**:
   - Offline mode: 0.7 * vector + 0.3 * BM25
   - Online mode: 0.15 * vector + 0.15 * BM25 + 0.7 * web
4. **Deduplication** and **Reranking**
5. Return top-5 documents

**Note:** Web search only active in online mode with TAVILY_API_KEY

---

### 4. Prepare Messages

**File:** `src/nodes/agent_response.py`

**Purpose:** Build context-aware messages for LLM

**Input:** `retrieved_docs` + `conversation_history` + `user_input`

**Output:** `messages` (list of SystemMessage, HumanMessage)

**Processing:**
1. Build system message with:
   - Agent persona and instructions
   - Retrieved documentation (if any)
   - Conversation history (last N turns)
2. Build human message with current query
3. Format as LangChain message objects

---

### 5. Agent

**File:** `src/llm/llm_nodes.py`

**Purpose:** Generate response using LLM

**Input:** `messages` (prepared context)

**Output:** `messages` (with AI response appended)

**Processing:**
1. Get cached LLM instance (intermediate model)
2. Invoke LLM with prepared messages
3. No tool calling - direct response generation
4. Append AI response to messages

**LLM Model:** Intermediate (gemini-2.5-flash)

---

### 6-10. Conversation Flow Nodes

**finalize**: Extracts final response text from AI message
**user_output**: Displays response to user (supports streaming)
**save_turn**: Saves query-response pair to conversation memory
**prompt_continue**: Asks user if they want to continue
**reset_for_next_turn**: Clears per-query state, preserves history

---

## State Management

### AgentState Schema

```python
# File: src/graph/state.py

class AgentState(TypedDict):
    # User input
    user_input: str

    # Security
    security_check: dict  # {is_safe: bool, reason: str}

    # Intent classification
    intent_result: IntentClassificationResult

    # RAG retrieval
    retrieved_docs: List[Document]

    # LLM interaction
    messages: List[BaseMessage]
    final_response: str

    # Conversation memory
    conversation_history: List[ConversationTurn]
    turn_count: int

    # Control flow
    continue_conversation: bool
```

### Conversation Memory

**Storage:** In-memory list of `ConversationTurn` objects

**Structure:**
```python
class ConversationTurn:
    query: str
    response: str
    intent: IntentClassificationResult
    retrieved_docs: List[str]  # Document titles
    timestamp: datetime
```

**Usage:**
- Maintains last `MAX_HISTORY_TURNS` (default: 5) turns
- Injected into context for history-aware responses
- Enables follow-up questions and clarifications

---

## Key Features

### 1. LLM Caching

**Implementation:** Global cache dictionary in `src/llm/llm.py`

**Models Cached:**
- `gemini-2.5-flash-lite` (fast)
- `gemini-2.5-flash` (intermediate)
- `gemini-2.5-pro` (slow)

**Benefits:**
- No LLM recreation on every invocation
- Reduces latency by ~200-500ms per query
- Memory efficient (only 3 instances)

---

### 2. Parallel Initialization

**Implementation:** Threading in `src/graph/initialization.py`

**Components Loaded:**
- LLM cache (3 models in parallel)
- RAG components (PKL files + VectorDB)
- Security model (ProtectAI DeBERTa v3)

**Benefits:**
- Faster startup (3-5 seconds vs 10-15 seconds serial)
- Idempotent (safe to call multiple times)
- Thread-safe (uses lock)

---

### 3. Hybrid RAG

**Vector Search:**
- ChromaDB with HNSW index
- sentence-transformers/all-MiniLM-L6-v2 embeddings
- Semantic similarity matching

**BM25 Search:**
- Keyword-based exact term matching
- Handles technical terms well (e.g., "StateGraph", "checkpointer")

**Scoring:**
- Offline: 70% vector + 30% BM25
- Online: 15% vector + 15% BM25 + 70% web

---

### 4. No User Tools

**Previous Design:** Agent could call `ask_user` tool for clarifications

**Current Design:**
- Intent classification detects clarification needs
- Conversation flow handles follow-ups naturally
- No tool calling in agent node (simpler, faster)

**Benefits:**
- Reduced complexity
- Faster responses (no tool invocation overhead)
- Cleaner conversation flow

---

### 5. Preprocessing Merged

**Previous Design:** Separate `preprocessing` node

**Current Design:** Preprocessing logic inside `intent_classification` node

**Benefits:**
- Fewer LLM calls (1 instead of 2)
- Reduced graph complexity
- Faster processing

---

## Performance Metrics

| Metric | Target | Actual |
|--------|--------|--------|
| Startup time (cold) | < 10s | ~5-7s |
| Startup time (warm) | < 2s | ~1-2s |
| Query latency (offline) | < 3s | ~2-3s |
| Query latency (online) | < 5s | ~4-5s |
| Retrieval latency | < 500ms | ~200-300ms |
| Memory usage | < 1GB | ~850MB |

---

## Configuration

### Environment Variables

See [settings.py](../settings.py) for all configuration options.

**Key Settings:**
- `SECURITY_ENABLED`: Enable/disable security gateway (default: false)
- `AGENT_MODE`: "offline" or "online" (default: offline)
- `MODEL_FAST`: Fast model tier (default: gemini-2.5-flash-lite)
- `MODEL_INTERMEDIATE`: Intermediate tier (default: gemini-2.5-flash)
- `MODEL_SLOW`: Slow tier (default: gemini-2.5-pro)
- `MAX_HISTORY_TURNS`: Conversation history length (default: 5)

---

## Troubleshooting

### Slow Startup

**Cause:** Large models or slow network

**Solution:**
- Check logs for which component is slow
- Ensure models are cached locally
- Verify network connection for first-time downloads

### High Memory Usage

**Cause:** Multiple LLM instances or large vector DB

**Solution:**
- Use smaller embedding model
- Reduce `MAX_HISTORY_TURNS`
- Enable lazy initialization

### Security Gateway Disabled

**Cause:** `SECURITY_ENABLED=false` in settings

**Solution:**
- Set `SECURITY_ENABLED=true`
- Install `transformers` package
- Download ProtectAI DeBERTa v3 model

---

**Questions?** See [DEV_GUIDE.md](DEV_GUIDE.md) for development details or [README.md](../README.md) for usage examples.
