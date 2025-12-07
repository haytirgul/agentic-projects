# Code RAG Agent - Agent Architecture

**Version:** 2.0
**Author:** Code RAG Agent Team
**Last Updated:** 2025-12-06

---

## Executive Summary

This document specifies the agent architecture for the Code RAG Agent—a LangGraph-based retrieval-augmented generation system for code understanding. The architecture implements a **state machine workflow** with **6 specialized nodes** that work together to provide secure, accurate, and context-aware responses.

### Design Philosophy

The agent layer is the **orchestration backbone** of the system. Architectural decisions here impact latency, reliability, and extensibility. This design prioritizes:

1. **Security First**: Prompt injection detection before any LLM processing
2. **Fast Path Optimization**: Skip expensive LLM calls when regex suffices
3. **Graceful Degradation**: Graduated fallbacks prevent zero-result failures
4. **Stateless Nodes**: Pure functions enable testing, debugging, and parallelization
5. **Streaming Support**: Real-time response generation for better UX

---

## Architecture Overview (v1.3)

```
                              +------------------+
                              |   User Query     |
                              +--------+---------+
                                       |
                                       v
+===========================================================================+
|                        AGENT LAYER (LangGraph)                            |
|                                                                           |
|  +------------------+     +-------------------+                           |
|  | Security Gateway |---->| Input Preprocessor|                           |
|  |  (ProtectAI ML)  |     |  (Regex + LLM)    |                           |
|  +--------+---------+     +--------+----------+                           |
|           |                        |                                      |
|           v (blocked)              |                                      |
|          END                       |                                      |
|                    +---------------+---------------+                      |
|                    |               |               |                      |
|                    v               v               v                      |
|             history_req      follow_up      needs_retrieval               |
|                    |        (no retrieval)        |                       |
|                    |               |              v                       |
|                    |               |    +--------------------+            |
|                    |               |    |       Router       |            |
|                    |               |    |  (Fast + LLM)      |            |
|                    |               |    +---------+----------+            |
|                    |               |              |                       |
|                    |               |              v                       |
|                    |               |    +--------------------+            |
|                    |               |    |     Retrieval      |            |
|                    |               |    |  (BM25 + FAISS)    |            |
|                    |               |    +---------+----------+            |
|                    |               |              |                       |
|                    |               +------+-------+                       |
|                    |                      |                               |
|                    |                      v                               |
|                    |            +--------------------+                    |
|                    |            |     Synthesis      |                    |
|                    |            |   (w/ Streaming)   |                    |
|                    |            +---------+----------+                    |
|                    |                      |                               |
|                    +----------+-----------+                               |
|                               |                                           |
|                               v                                           |
|                    +--------------------+                                 |
|                    | Conversation Memory|                                 |
|                    +--------------------+                                 |
|                               |                                           |
|                               v                                           |
|                              END                                          |
+===========================================================================+
```

**Key Paths (v1.3):**
- `history_request` → Skip to conversation_memory (return history summary)
- `follow_up` (history sufficient) → Skip to synthesis (no retrieval needed)
- `follow_up_with_retrieval` / `new_question` → Router → Retrieval → Synthesis

---

## Design Decisions

### Decision 1: LangGraph vs LangChain Chains

**Problem:** How to orchestrate the multi-stage RAG pipeline?

**Options Evaluated:**

| Approach | Conditional Routing | State Management | Debugging | Extensibility |
|----------|--------------------|-----------------| ----------|---------------|
| Linear Chain | ❌ No | ❌ Implicit | ❌ Hard | ❌ Rigid |
| LCEL (RunnablePassthrough) | ⚠️ Limited | ⚠️ Manual | ⚠️ Moderate | ⚠️ Moderate |
| **LangGraph StateGraph** | ✅ Native | ✅ TypedDict | ✅ Trace | ✅ **Excellent** |

**Decision:** LangGraph StateGraph with TypedDict state schema.

**Rationale:**
- Conditional edges enable security-based early termination
- Explicit state schema prevents "magic dictionary" anti-pattern
- Native checkpointing for conversation persistence
- Graph visualization for debugging complex flows

**Trade-off Accepted:** LangGraph adds learning curve vs simple chains. Justified by workflow complexity.

**Implementation:** [graph.py](../src/agent/graph.py)

---

### Decision 2: Security Gateway Position

**Problem:** Where in the pipeline to validate user input?

**Options Evaluated:**

| Position | Latency Impact | Security Coverage | Cost Efficiency |
|----------|---------------|-------------------|-----------------|
| After LLM router | Saves security latency on valid queries | ❌ LLM sees malicious input | ❌ Wasted LLM calls |
| **Before any LLM** | +50-100ms always | ✅ Full protection | ✅ **Blocks before LLM** |
| Async (parallel) | Minimal | ⚠️ Race condition risk | ⚠️ Complex error handling |

**Decision:** Security Gateway as the **first node** in the pipeline.

**Rationale:**
- Malicious queries never reach the LLM (security boundary)
- Saves LLM costs on blocked requests
- Simple fail-fast architecture
- 50-100ms overhead acceptable for security guarantee

**Implementation:** [security_gateway.py](../src/agent/nodes/security_gateway.py)

---

### Decision 3: Hybrid Intent Classification (Input Preprocessor)

**Problem:** How to detect if user query references conversation history or previous context?

**Options Evaluated:**

| Approach | Latency | Accuracy | Cost |
|----------|---------|----------|------|
| LLM for all queries | ~500ms always | High | High |
| Regex only | <1ms | Low (misses implicit refs) | Zero |
| **Hybrid (Regex → LLM fallback)** | <1ms or ~500ms | **High** | **Optimized** |

**Decision:** Two-stage hybrid classification with **four intent types**:
1. **Fast path:** Regex patterns for explicit history queries (<1ms)
2. **LLM fallback:** Intent classification only when history exists (~500ms)

**Intent Types (v1.3):**

| Intent | Description | Needs Retrieval | Example |
|--------|-------------|-----------------|---------|
| `history_request` | User asks about conversation history | ❌ No | "What did I ask earlier?" |
| `follow_up` | References history AND history is sufficient | ❌ No | "Can you explain that simpler?" |
| `follow_up_with_retrieval` | References history BUT needs new info | ✅ Yes | "Also show me the tests for that" |
| `new_question` | Independent question | ✅ Yes | "How does BM25 work?" |

**Algorithm:**

```
Query: "Also show me the error handling for that class"
         |
         v
+---------------------------+
| Stage 1: Regex Patterns   |  <-- Matches explicit history patterns
| "what did I ask/say..."   |
+---------------------------+
         | Match? → Return history response (needs_retrieval=False)
         | No match
         v
+---------------------------+
| Stage 2: LLM Classification|  <-- Only if conversation_history exists
| (Uses structured output)   |
+---------------------------+
         |
         ├── history_request → Return history summary (needs_retrieval=False)
         ├── follow_up → Resolve references, skip to synthesis (needs_retrieval=False)
         ├── follow_up_with_retrieval → Resolve refs, go to router (needs_retrieval=True)
         └── new_question → Clean query, go to router (needs_retrieval=True)
```

**Key Decision: `follow_up` vs `follow_up_with_retrieval`**

This distinction is critical for handling queries like:
- "Tell me more about that" → `follow_up` (history has the answer)
- "What about the tests for that?" → `follow_up_with_retrieval` (tests not in history)

The LLM determines if the previous answer contains enough information to answer the new query, or if fresh retrieval is needed.

**Rationale:**
- 80% of queries are new questions (no LLM call needed)
- LLM only invoked for ambiguous queries with existing history
- Structured output ensures consistent classification
- `needs_retrieval` flag prevents unnecessary LLM calls when history suffices

**Implementation:** [input_preprocessor.py](../src/agent/nodes/input_preprocessor.py)

---

### Decision 4: Router Fast Path Optimization

**Problem:** LLM routing takes ~1.5s. Can we skip it for simple queries?

**Options Evaluated:**

| Approach | Latency | Coverage | Maintenance |
|----------|---------|----------|-------------|
| LLM for all queries | ~1.5s always | 100% | Low |
| Regex only | <10ms | ~30% (limited patterns) | Medium |
| **Regex fast path + LLM fallback** | <10ms or ~1.5s | **100%** | **Medium** |

**Decision:** Fast path router with regex patterns for common query types.

**Fast Path Patterns:**

| Pattern | Regex | Example | Hit Rate |
|---------|-------|---------|----------|
| Hex codes | `^0x[0-9A-Fa-f]+$` | `0x884` | ~5% |
| Function names | `^[a-z_][a-z0-9_]*$` | `get_user` | ~10% |
| Class names | `^[A-Z][a-zA-Z0-9]*$` | `HTTPClient` | ~5% |
| File names | `^[\w\-]+\.[a-z]{2,4}$` | `config.toml` | ~5% |

**Expected Results:**
- ~20-25% of queries hit fast path (<10ms)
- Remaining queries use LLM router (~1.5s)
- Average latency reduction: ~300-400ms

**Implementation:** [fast_path_router.py](../src/retrieval/fast_path_router.py)

---

### Decision 5: Stateless Node Functions

**Problem:** How to structure graph nodes for testability and parallelization?

**Options Evaluated:**

| Approach | Testability | Side Effects | Parallelization |
|----------|------------|--------------|-----------------|
| Class methods with instance state | ⚠️ Requires mocking | ❌ Hidden state | ❌ Race conditions |
| **Pure functions** | ✅ Simple fixtures | ✅ Explicit I/O | ✅ **Safe** |
| Coroutines with shared state | ⚠️ Async complexity | ⚠️ Lock management | ⚠️ Deadlock risk |

**Decision:** All nodes are **pure functions** with signature `(state: AgentState) -> dict[str, Any]`.

**Pattern:**

```python
def node_name(state: AgentState) -> dict[str, Any]:
    """
    Node description.

    Reads: state.field1, state.field2
    Writes: output_field1, output_field2
    """
    # Read from state
    input_value = state.get("field1")

    # Process (pure computation)
    output_value = process(input_value)

    # Return state updates (not mutations)
    return {
        "output_field1": output_value,
        "output_field2": derived_value,
    }
```

**Benefits:**
- ✅ Unit testing with simple dict fixtures
- ✅ No hidden state mutations
- ✅ Easy to reason about data flow
- ✅ Enables future parallelization

**Implementation:** All files in [src/agent/nodes/](../src/agent/nodes/)

---

### Decision 6: Streaming Synthesis Output

**Problem:** Synthesis takes 2-3s. Should we stream tokens to the user?

**Options Evaluated:**

| Approach | User Experience | Implementation | Error Handling |
|----------|----------------|----------------|----------------|
| Wait for complete response | Poor (blank screen) | Simple | Simple |
| **Stream tokens** | **Excellent** (progressive) | Medium | Requires buffering |
| Chunk-based streaming | Good | Complex | Complex |

**Decision:** Stream tokens using LLM's native `.stream()` method.

**Implementation:**

```python
def synthesis_node(state: AgentState) -> dict[str, Any]:
    llm = get_cached_llm(SYNTHESIS_MODEL)
    messages = build_synthesis_messages(...)

    answer_chunks = []
    for chunk in llm.stream(messages):
        content = chunk.content
        if content:
            answer_chunks.append(content)
            sys.stdout.write(content)  # Real-time output
            sys.stdout.flush()

    return {"final_answer": "".join(answer_chunks)}
```

**Rationale:**
- Perceived latency drops from 2-3s to ~200ms (first token)
- Users can start reading while generation continues
- Natural fit for chat interfaces

**Implementation:** [synthesis.py](../src/agent/nodes/synthesis.py)

---

### Decision 7: State Schema Design

**Problem:** How to structure shared state across nodes?

**Options Evaluated:**

| Approach | Type Safety | IDE Support | Validation |
|----------|------------|-------------|------------|
| Plain dict | ❌ None | ❌ No autocomplete | ❌ Runtime errors |
| Pydantic BaseModel | ✅ Full | ✅ Full | ✅ At creation |
| **TypedDict (total=False)** | ✅ Static only | ✅ Full | ⚠️ Runtime optional |

**Decision:** TypedDict with `total=False` for optional fields.

**Rationale:**
- LangGraph expects TypedDict (native integration)
- `total=False` allows incremental state building
- IDE autocomplete for all field names
- Lighter than Pydantic (no validation overhead per update)

**Schema Structure:**

```python
class AgentState(TypedDict, total=False):
    # ═══ USER INPUT ═══
    user_query: str

    # ═══ SECURITY ═══
    gateway_passed: bool | None
    gateway_reason: str | None
    is_blocked: bool | None

    # ═══ INPUT PREPROCESSING (v1.3) ═══
    cleaned_query: str | None
    is_history_query: bool | None
    is_follow_up: bool | None
    needs_retrieval: bool | None  # v1.3: Whether fresh retrieval is needed

    # ═══ ROUTER ═══
    router_output: RouterOutput | None
    codebase_tree: str | None

    # ═══ RETRIEVAL ═══
    retrieved_chunks: list[dict] | None
    expanded_chunks: list[dict] | None
    retrieval_metadata: dict | None

    # ═══ SYNTHESIS ═══
    messages: list[BaseMessage]
    final_answer: str | None
    citations: list[dict] | None

    # ═══ CONVERSATION ═══
    conversation_history: list[dict] | None
    turn_count: int | None

    # ═══ ERROR ═══
    error: str | None
```

**Implementation:** [state.py](../src/agent/state.py)

---

## Graph Flow

### Complete Pipeline (v1.3)

```
START
   │
   ▼
┌─────────────────────────┐
│   1. Security Gateway   │  ← ML-based prompt injection detection
│   (ProtectAI DeBERTa)   │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │ Blocked?      │
    └───────┬───────┘
            │
   YES ◄────┴────► NO
    │               │
    ▼               ▼
   END    ┌─────────────────────────┐
          │  2. Input Preprocessor  │  ← Regex + LLM intent classification
          │  (Query understanding)  │
          └───────────┬─────────────┘
                      │
              ┌───────┴────────────────┐
              │ Intent classification   │
              └───────┬────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
history_req    follow_up (no       follow_up_with_retrieval
               retrieval)          OR new_question
    │                 │                 │
    ▼                 │                 ▼
┌────────────┐        │      ┌─────────────────────────┐
│  Return    │        │      │       3. Router         │
│  history   │        │      │  (Fast path + LLM)      │
│  summary   │        │      └───────────┬─────────────┘
└─────┬──────┘        │                  │
      │               │                  ▼
      │               │      ┌─────────────────────────┐
      │               │      │      4. Retrieval       │
      │               │      │   (BM25 + FAISS + RRF)  │
      │               │      └───────────┬─────────────┘
      │               │                  │
      │               └────────┬─────────┘
      │                        │
      │                        ▼
      │            ┌─────────────────────────┐
      │            │      5. Synthesis       │
      │            │   (Streaming + Cite)    │
      │            └───────────┬─────────────┘
      │                        │
      └──────────┬─────────────┘
                 │
                 ▼
      ┌─────────────────────────┐
      │  6. Conversation Memory │
      │   (Save turn, reset)    │
      └───────────┬─────────────┘
                  │
                  ▼
                 END
```

### Conditional Routing Logic

```python
# routing.py - Edge conditions (v1.3)

def route_after_security(state: AgentState) -> str:
    """Route based on security validation."""
    if state.get("is_blocked"):
        return END
    return "input_preprocessor"

def route_after_input_preprocessor(state: AgentState) -> str:
    """Route based on query intent and retrieval needs (v1.3).

    Three possible paths:
    1. history_request → conversation_memory (final_answer already set)
    2. follow_up (no retrieval) → synthesis (use history context)
    3. follow_up_with_retrieval / new_question → router (needs fresh data)
    """
    if state.get("is_history_query"):
        return "conversation_memory"  # History query shortcut

    if not state.get("needs_retrieval", True):
        return "synthesis"  # Follow-up with sufficient history

    return "router"  # New question or follow-up needing retrieval
```

**Key Change (v1.3):** The `follow_up` intent now skips retrieval when conversation history contains sufficient information to answer the query. This prevents unnecessary LLM calls when users ask for clarification or rephrasing.

**Implementation:** [routing.py](../src/agent/routing.py)

---

## Component Specifications

### File Structure

| Component | File | Responsibility |
|-----------|------|----------------|
| **Graph Builder** | [graph.py](../src/agent/graph.py) | Constructs and compiles LangGraph workflow |
| **State Schema** | [state.py](../src/agent/state.py) | TypedDict shared across all nodes |
| **Routing Logic** | [routing.py](../src/agent/routing.py) | Conditional edge functions |

### Graph Nodes

| Node | File | Latency | Purpose |
|------|------|---------|---------|
| `security_gateway` | [security_gateway.py](../src/agent/nodes/security_gateway.py) | 50-100ms | ML-based prompt injection detection |
| `input_preprocessor` | [input_preprocessor.py](../src/agent/nodes/input_preprocessor.py) | <1ms or ~500ms | Query cleaning and intent classification |
| `router` | [router.py](../src/agent/nodes/router.py) | <10ms or ~1.5s | Query decomposition with fast path |
| `retrieval` | [retrieval.py](../src/agent/nodes/retrieval.py) | 100-300ms | Hybrid search + context expansion |
| `synthesis` | [synthesis.py](../src/agent/nodes/synthesis.py) | 2-3s | Streaming answer generation |
| `conversation_memory` | [conversation_memory.py](../src/agent/nodes/conversation_memory.py) | <1ms | Turn management and state reset |

---

## Data Flow Example

### Query: "How does BM25 tokenization work?"

```
1. USER QUERY
   "How does BM25 tokenization work?"
        │
        ▼
2. SECURITY GATEWAY (50ms)
   ├── Input: "How does BM25 tokenization work?"
   ├── Model: ProtectAI DeBERTa v3
   ├── Result: SAFE (confidence: 0.02)
   └── Output: {gateway_passed: true}
        │
        ▼
3. INPUT PREPROCESSOR (<1ms)
   ├── Input: "How does BM25 tokenization work?"
   ├── Regex check: No history pattern match
   ├── History exists: No
   ├── Action: Simple cleaning
   └── Output: {cleaned_query: "BM25 tokenization work", is_follow_up: false}
        │
        ▼
4. ROUTER (1.5s - LLM path)
   ├── Input: "BM25 tokenization work"
   ├── Fast path: No match (complex query)
   ├── LLM router: Generate structured request
   └── Output: {
         router_output: RouterOutput(
           cleaned_query: "BM25 tokenization implementation",
           retrieval_requests: [
             RetrievalRequest(
               query: "BM25 tokenize split camelCase snake_case",
               source_types: ["code"],
               folders: ["src/retrieval/"],
               reasoning: "Need BM25 tokenization logic for code search"
             )
           ]
         )
       }
        │
        ▼
5. RETRIEVAL (200ms)
   ├── Metadata filter: 45 candidates in src/retrieval/
   ├── BM25 search: Top 50 by term frequency
   ├── FAISS search: Top 50 by vector similarity
   ├── Weighted RRF: BM25(0.4) + Vector(1.0)
   ├── Context expansion: Add parent class, sibling methods
   └── Output: {
         expanded_chunks: [5 chunks with context],
         retrieval_metadata: {bm25_count: 50, faiss_count: 50}
       }
        │
        ▼
6. SYNTHESIS (2.5s streaming)
   ├── Build prompt with expanded chunks
   ├── Include conversation history (empty)
   ├── Stream tokens to stdout
   └── Output: {
         final_answer: "BM25 tokenization in the codebase splits...",
         citations: [{file: "hybrid_retriever.py", line: 142}]
       }
        │
        ▼
7. CONVERSATION MEMORY (<1ms)
   ├── Save turn: {query, answer, citations, timestamp}
   ├── Increment turn_count: 1
   ├── Reset per-query state
   └── Output: {conversation_history: [turn_1], turn_count: 1}
        │
        ▼
8. END
   Return final_answer with citations
```

---

## Performance Characteristics

| Stage | Best Case | Typical | Worst Case | Notes |
|-------|-----------|---------|------------|-------|
| Security Gateway | 50ms | 75ms | 100ms | Batch classification |
| Input Preprocessor | <1ms | <1ms | 500ms | LLM only if history exists |
| Router | <10ms | 1.5s | 2s | Fast path vs LLM |
| Retrieval | 100ms | 200ms | 500ms | Depends on corpus size |
| Synthesis | 2s | 2.5s | 4s | LLM generation |
| Conversation Memory | <1ms | <1ms | <1ms | In-memory operations |

**End-to-End Latency:**
- **Fast path (20% of queries):** ~2.5-3s
- **LLM path (80% of queries):** ~4-5s
- **First token (streaming):** ~200ms

---

## Error Handling Strategy

### Graceful Degradation

```python
# Pattern used across all nodes

def node_with_fallback(state: AgentState) -> dict[str, Any]:
    try:
        # Primary logic
        result = primary_operation()
        return {"result": result}
    except SpecificException as e:
        # Recoverable error - use fallback
        logger.warning(f"Primary failed, using fallback: {e}")
        fallback_result = fallback_operation()
        return {"result": fallback_result}
    except Exception as e:
        # Unrecoverable - propagate error in state
        logger.error(f"Node failed: {e}", exc_info=True)
        return {"error": f"Node failed: {str(e)}"}
```

### Fallback Chain

| Component | Primary | Fallback 1 | Fallback 2 | Emergency |
|-----------|---------|------------|------------|-----------|
| Security | ML classification | Fail-open (allow) | - | - |
| Router | LLM decomposition | Fast path patterns | Single broad query | - |
| Metadata Filter | Strict filters | Drop file_patterns | Drop folders | Search all |
| Retrieval | Hybrid (BM25+FAISS) | BM25 only | Vector only | Return empty |

---

## Extension Points

| Extension | Location | How to Extend |
|-----------|----------|---------------|
| **Add new node** | `src/agent/nodes/` | Create function, add to `graph.py` |
| **New routing condition** | `routing.py` | Add conditional edge function |
| **Different LLM provider** | `src/llm/llm.py` | Swap ChatGoogleGenerativeAI |
| **Custom security model** | `src/security/` | Replace PromptGuardClassifier |
| **New retrieval source** | `src/retrieval/` | Extend MetadataFilter, ChunkLoader |
| **Additional state fields** | `state.py` | Add to AgentState TypedDict |

---

## References

- **Agent Architecture**: This document
- **Retrieval Architecture**: [RETRIEVAL_ARCHITECTURE.md](RETRIEVAL_ARCHITECTURE.md)
- **Chunking Architecture**: [CHUNKING_ARCHITECTURE.md](CHUNKING_ARCHITECTURE.md)
- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
