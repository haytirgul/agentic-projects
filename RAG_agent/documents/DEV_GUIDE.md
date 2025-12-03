# Developer Guide - Opsfleet Documentation Assistant

**Version**: 3.0
**Last Updated**: 2025-11-30
**Audience**: Developers and AI Agents working on this codebase

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Key Design Patterns](#key-design-patterns)
6. [Development Workflow](#development-workflow)
7. [Testing Strategy](#testing-strategy)
8. [Extending the System](#extending-the-system)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Required API Keys

Add these to `.env`:

```bash
GOOGLE_API_KEY=your_google_api_key          # Get from https://aistudio.google.com/app/apikey
TAVILY_API_KEY=your_tavily_key              # Optional, for online mode: https://tavily.com
AGENT_MODE=offline                           # offline | online
LOG_LEVEL=INFO                               # DEBUG | INFO | WARNING | ERROR
```

### Running the Agent

```bash
# Build RAG index (first time only)
python scripts/data_pipeline/indexing/build_vector_index.py

# Run chatbot
python chatbot.py
```

---

## Project Structure

```
rag_agent/
├── chatbot.py                     # Interactive CLI chatbot
├── settings.py                    # Centralized configuration
├── requirements.txt               # Python dependencies
│
├── src/                           # Core application logic
│   ├── graph/                     # LangGraph workflow
│   │   ├── builder.py             # Graph definition and compilation
│   │   ├── routing.py             # Edge routing logic
│   │   └── state.py               # State management schemas
│   │
│   ├── nodes/                     # Graph node implementations
│   │   ├── preprocessing.py       # Request parsing and validation
│   │   ├── intent_classification.py  # Intent analysis agent
│   │   ├── hybrid_retrieval.py    # RAG retrieval (VectorDB + BM25 + Fuzzy)
│   │   └── agent_response.py      # Response generation
│   │
│   ├── rag/                       # RAG components
│   │   ├── vector_search.py       # ChromaDB vector search
│   │   ├── bm25_search.py         # BM25 keyword search
│   │   └── web_search_api.py      # Web search integration (online mode)
│   │
│   ├── llm/                       # LLM client and operations
│   │   ├── llm.py                 # LLM initialization
│   │   └── operations.py          # Common LLM operations
│   │
│   └── tools/                     # Agent tools
│       └── ask_user.py            # Human-in-the-loop tool
│
├── models/                        # Pydantic data models
│   ├── request_preprocessing.py   # Request parsing models
│   ├── intent_classification.py   # Intent classification models
│   ├── document.py                # Document models (in-memory + retrieved)
│   └── conversation_memory.py     # Conversation context models
│
├── prompts/                       # Prompt templates (as code)
│   ├── request_parser.py          # Request parsing prompt
│   ├── intent_classification.py   # Intent classification prompt
│   ├── agent_response.py          # Response generation prompt
│   ├── web_search_query_generation.py  # Web search query generation
│   └── examples.py                # Prompt examples
│
├── data/                          # Data directory
│   ├── input/                     # Raw documentation (llms.txt)
│   ├── output/                    # Processed data
│   │   ├── json_files/            # Parsed JSON docs (547 files)
│   │   │   ├── langchain/
│   │   │   ├── langgraph/
│   │   │   └── langsmith/
│   │   └── chroma_db/             # Vector database
│   └── cache/                     # Embedding cache
│
├── data_processing/               # Data ingestion scripts
│   ├── build_rag_index.py         # Main script to build vector DB
│   ├── langchain_parser.py        # LangChain docs parser
│   └── langgraph_parser.py        # LangGraph docs parser
│
├── scripts/                       # Utility scripts
│   ├── build_vector_index.py      # Rebuild vector index
│   ├── test_vector_search.py      # Test vector search
│   └── download_docs.py           # Download latest llms.txt files
│
├── tests/                         # Test suite
│   ├── test_graph.py              # Graph integration tests
│   ├── test_intent_classification.py  # Intent classification tests
│   ├── test_hybrid_retrieval.py   # Hybrid retrieval tests
│   └── evaluation/                # Evaluation datasets
│       └── test_queries.py        # 50 evaluation queries
│
└── documents/                     # Project documentation
    ├── DEV_GUIDE.md               # This file
    ├── README.md                  # User-facing documentation
    ├── ARCHITECTURE.md            # System architecture
    ├── RAG_IMPLEMENTATION_PRD.md  # RAG implementation details
    └── task.md                    # Original assignment
```

---

## Core Components

### 1. LangGraph Workflow (`src/graph/builder.py`)

The agent is built as a **state machine** using LangGraph. The graph orchestrates the flow from user input to final response.

**Graph Structure:**

```
START
  ↓
preprocessing (parse + validate)
  ↓
intent_classification_agent (analyze intent)
  ↓ (tool loop if clarification needed)
intent_classification_tools ──┐
  ↓                            │
  └────────────────────────────┘
  ↓
intent_classification_finalize (extract structured result)
  ↓
hybrid_retrieval (VectorDB + BM25 + Fuzzy search)
  ↓
prepare_messages (inject docs into context)
  ↓
agent (generate response with citations)
  ↓ (tool loop if ask_user called)
tools ──┐
  ↓     │
  └─────┘
  ↓
finalize (extract final response)
  ↓
END
```

**Key Files:**
- `src/graph/builder.py` - Graph construction
- `src/graph/routing.py` - Conditional edge routing
- `src/graph/state.py` - State schema (`AgentState`)

### 2. Preprocessing Node (`src/nodes/preprocessing.py`)

**Purpose**: Clean and validate user input before processing.

**Operations:**
1. **Request Parsing** (LLM call):
   - Remove greetings and filler words
   - Extract code snippets
   - Extract structured data (errors, configs)
   - Normalize text

2. **Security Validation** (LLM call):
   - Check for prompt injection
   - Check for harmful content
   - Check relevance to LangChain/LangGraph/LangSmith

**Output:**
- `cleaned_request`: Cleaned text
- `code_snippets`: Extracted code blocks
- `gateway_result`: Security validation result

### 3. Intent Classification Node (`src/nodes/intent_classification.py`)

**Purpose**: Understand what the user wants to optimize RAG retrieval.

**Intent Types:**
- `factual_lookup` - Quick facts (e.g., "What is a checkpointer?")
- `implementation_guide` - How-to code examples
- `troubleshooting` - Error debugging
- `conceptual_explanation` - Deep understanding
- `best_practices` - Recommendations
- `comparison` - X vs Y
- `configuration_setup` - Installation/config
- `migration` - Version upgrades
- `capability_exploration` - What's possible
- `api_reference` - API docs

**Output:**
- `intent_result`: Structured `IntentClassification` object with:
  - `framework`: langchain | langgraph | langsmith | general
  - `language`: python | javascript | both
  - `keywords`: List of extracted keywords
  - `topics`: List of relevant topics
  - `intent_type`: One of the types above
  - `confidence`: 0-1 confidence score

**Uses Agent Loop**: If the request is vague, the agent calls `ask_user` tool to get clarification before finalizing the intent.

### 4. Hybrid Retrieval Node (`src/nodes/hybrid_retrieval.py`)

**Purpose**: Retrieve top-5 most relevant documents using hybrid search.

**Hybrid Scoring Formula:**

```python
final_score = (
    0.4 * bm25_normalized_score +      # Keyword matching
    0.3 * vector_normalized_score +     # Semantic similarity
    0.2 * fuzzy_normalized_score +      # Typo tolerance
    0.1 * intent_boost_score            # Framework/language match bonus
)
```

**Retrieval Methods:**

1. **VectorDB Search** (`src/rag/vector_search.py`):
   - Uses ChromaDB with Google Gemini embeddings
   - Semantic similarity search
   - Filters by framework/language/topic
   - Returns top-30 results

2. **BM25 Search** (`src/rag/bm25_search.py`):
   - Keyword-based search using `rank-bm25`
   - Built over all document content
   - Returns top-50 results

**Parallel Execution**: All three searches run concurrently using `ThreadPoolExecutor` for ~1.8x speedup.

**Output:**
- `retrieved_documents`: List of top-5 `RetrievedDocument` objects
- `retrieval_metadata`: Debug info (scores, query, etc.)

### 5. Response Generation Node (`src/nodes/agent_response.py`)

**Purpose**: Generate answer with citations using agent tools.

**Process:**
1. Build system prompt with intent context
2. Inject retrieved documents into user message
3. Call LLM with agent tools (ask_user for clarifications)
4. Extract final response

**Tools Available:**
- `ask_user`: Ask for clarification if retrieved docs are insufficient

**Output:**
- `final_response`: Generated answer with citations

---

## Data Flow

### Example: Clear Question (No Clarification)

```
User: "How do I create a LangGraph StateGraph with persistence?"

[1] preprocessing
    Input:  "How do I create a LangGraph StateGraph with persistence?"
    Output: cleaned_request, gateway_result=valid

[2] intent_classification_agent
    Output: IntentClassification(
        intent_type="implementation_guide",
        framework="langgraph",
        language="python",
        topics=["StateGraph", "persistence", "checkpointing"],
        needs_clarification=False
    )

[3] hybrid_retrieval
    - VectorDB search: 30 results
    - BM25 search: 50 results
    - Fuzzy search: 20 results
    - Hybrid scoring → Top-5 documents

    Output: retrieved_documents=[
        Document(title="Persistence", file_path="langgraph/python/concepts/persistence.json", score=85.2),
        Document(title="StateGraph", file_path="langgraph/python/how-tos/state-graph-tutorial.json", score=78.3),
        ...
    ]

[4] prepare_messages
    Injects top-5 docs into context with citations

[5] agent
    Generates response using retrieved docs

[6] finalize
    Output: final_response with citations

Total LLM Calls: 3 (preprocessing=2, intent=1)
Total Latency: ~2-3s
```

### Example: Vague Question (With Clarification)

```
User: "Help me with agents"

[1-2] preprocessing + intent_classification_agent
    Output: IntentClassification(confidence=0.55, vague=True)

[3] agent calls ask_user tool
    Agent: "I need clarification. Are you asking about LangChain or LangGraph agents? Which programming language?"

[User provides clarification: "LangGraph agents in Python"]

[4] intent_classification_agent (re-run with clarification)
    Output: IntentClassification(
        framework="langgraph",
        language="python",
        confidence=0.85
    )

[5-7] hybrid_retrieval + prepare_messages + agent
    Proceeds normally with clarified intent

Total LLM Calls: 4 (preprocessing=2, intent=1, re-intent=1)
```

---

## Key Design Patterns

### 1. Prompt-as-Code

All prompts are **Python functions** in the `prompts/` directory, not inline strings.

**Benefits:**
- Version control for prompts
- Easy testing and iteration
- Dynamic context injection
- Separation of concerns

**Example:**

```python
# prompts/intent_classification.py

def get_intent_classification_prompt(cleaned_request: str, keywords: List[str]) -> str:
    return f"""
You are an expert at classifying user intent for LangChain/LangGraph documentation questions.

User Request: {cleaned_request}
Extracted Keywords: {", ".join(keywords)}

Classify the intent type and extract metadata.

Output as JSON following the IntentClassification schema.
"""
```

### 2. Structured Outputs with Pydantic

**All LLM outputs** are validated against Pydantic models.

**Example:**

```python
# models/intent_classification.py

class IntentClassification(BaseModel):
    intent_type: Literal[
        "factual_lookup",
        "implementation_guide",
        "troubleshooting",
        # ... others
    ]
    framework: Optional[Literal["langchain", "langgraph", "langsmith", "general"]]
    language: Optional[Literal["python", "javascript", "both"]]
    keywords: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
```

**LLM Call:**

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=IntentClassification)
result = llm.invoke(prompt)
intent = parser.parse(result.content)
# intent is now a validated IntentClassification object
```

### 3. Hybrid Search Pattern

**Don't rely on a single retrieval method.** Combine multiple signals:

1. **VectorDB**: Handles semantic similarity, synonyms, paraphrases
2. **BM25**: Handles exact keyword matching, technical terms
3. **Fuzzy**: Handles typos, variations

**Key Insight**: Each method has strengths and weaknesses. Hybrid scoring leverages all strengths.

### 4. Agent Tools for Human-in-the-Loop

When the agent needs more information, it **doesn't guess** - it asks the user.

**Implementation:**

```python
# src/tools/ask_user.py

from langchain_core.tools import tool

@tool
def ask_user(question: str) -> str:
    """
    Ask the user a clarifying question.

    Args:
        question: The question to ask the user

    Returns:
        User's response
    """
    # This triggers an interrupt in LangGraph
    # The graph will pause and wait for user input
    raise GraphInterrupt(question)
```

**Usage in Graph:**

```python
# The agent calls the tool
agent_result = agent.invoke(state)

# If ask_user was called, the graph pauses
# User provides answer
# Graph resumes with user's response
```

### 5. Fail-Fast Validation

**Security checks happen early** in the preprocessing node, not later.

**Benefits:**
- Stop prompt injection before expensive operations
- Clear rejection messages
- Reduced cost (no wasted LLM calls)

---

## Development Workflow

### Adding a New Node

1. **Create node function** in `src/nodes/`:

```python
# src/nodes/my_new_node.py

from typing import Dict, Any

def my_new_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Description of what this node does.

    Input state:
        - required_field: Description

    Output state:
        - output_field: Description
    """
    # Node logic here
    return {"output_field": result}
```

2. **Add node to graph** in `src/graph/builder.py`:

```python
from src.nodes.my_new_node import my_new_node

def build_agent_graph():
    graph = StateGraph(AgentState)

    # Add node
    graph.add_node("my_new_node", my_new_node)

    # Add edges
    graph.add_edge("previous_node", "my_new_node")
    graph.add_edge("my_new_node", "next_node")

    return graph
```

3. **Write tests** in `tests/test_my_new_node.py`:

```python
def test_my_new_node():
    state = {"required_field": "test_value"}
    result = my_new_node(state)
    assert "output_field" in result
```

### Adding a New Prompt

1. **Create prompt function** in `prompts/`:

```python
# prompts/my_prompt.py

def get_my_prompt(context: str) -> str:
    return f"""
You are an expert at...

Context: {context}

Task: Do something specific

Output: JSON schema
"""
```

2. **Use in node**:

```python
from prompts.my_prompt import get_my_prompt

def my_node(state):
    prompt = get_my_prompt(state["context"])
    result = llm.invoke(prompt)
    return {"result": result}
```

### Adding a New Intent Type

1. **Update model** in `models/intent_classification.py`:

```python
class IntentClassification(BaseModel):
    intent_type: Literal[
        "factual_lookup",
        # ... existing types ...
        "new_intent_type",  # Add here
    ]
```

2. **Update prompt** in `prompts/intent_classification.py` to describe the new type

3. **Update RAG logic** if needed (e.g., different scoring for this intent)

---

## Testing Strategy

### Unit Tests

Test individual components in isolation:

```bash
# Test a specific node
pytest tests/test_intent_classification.py::test_intent_classification_node -v

# Test hybrid retrieval
pytest tests/test_hybrid_retrieval.py -v
```

### Integration Tests

Test full graph execution:

```bash
# Test complete flow
pytest tests/test_graph.py -v
```

### Evaluation Queries

Measure quality on 50 test queries:

```bash
# Run evaluation
pytest tests/evaluation/test_queries.py -v

# Measure Precision@5
python tests/evaluation/measure_precision.py
```

### Manual Testing

```bash
# Interactive chatbot testing
python chatbot.py
```

---

## Extending the System

### Adding a New Retrieval Method

Example: Add fuzzy semantic search

1. **Create retriever** in `src/rag/fuzzy_semantic_search.py`:

```python
class FuzzySemanticSearch:
    def __init__(self, documents):
        # Initialize

    def search(self, query, top_k=20):
        # Search logic
        return results
```

2. **Integrate into hybrid retrieval**:

```python
# src/nodes/hybrid_retrieval.py

_fuzzy_semantic = FuzzySemanticSearch(all_docs)

def hybrid_retrieval_node(state):
    # Run all searches in parallel
    fuzzy_semantic_results = _fuzzy_semantic.search(query)

    # Update scoring formula
    final_score = (
        0.3 * bm25_score +
        0.3 * vector_score +
        0.2 * fuzzy_score +
        0.1 * fuzzy_semantic_score +  # New method
        0.1 * intent_boost
    )
```

3. **Benchmark** - measure impact on Precision@5

### Adding Web Search (Online Mode)

Web search integration is planned in `RAG_IMPLEMENTATION_PRD.md` Appendix E.

**Key steps:**
1. Add `AGENT_MODE=online` environment variable
2. Integrate Tavily API
3. Run web search in parallel with offline retrieval
4. Deduplicate against offline docs
5. Mix web + offline results in top-5

**See**: `documents/RAG_IMPLEMENTATION_PRD.md` for full implementation plan.

---

## Troubleshooting

### Graph Compilation Errors

**Error**: `ValueError: Node X is not defined`

**Solution**: Make sure all nodes added to the graph are defined:

```python
graph.add_node("my_node", my_node_function)  # my_node_function must exist
```

### State Schema Mismatches

**Error**: `KeyError: 'field_name'`

**Solution**: Check `AgentState` schema in `src/graph/state.py`. All state fields must be defined:

```python
class AgentState(TypedDict, total=False):
    my_field: Optional[str]  # Add missing field
```

### LLM Output Parsing Failures

**Error**: `OutputParserException: Failed to parse output`

**Solution**:
1. Check the prompt - is it clear about output format?
2. Add examples to the prompt
3. Use `PydanticOutputParser` for validation
4. Add retry logic with exponential backoff

### Vector Search Returns No Results

**Problem**: Vector search always returns empty list

**Debug steps**:

```python
# Check if ChromaDB has data
collection = client.get_collection("langgraph_docs")
print(collection.count())  # Should be > 0

# If 0, rebuild index
python scripts/data_pipeline/indexing/build_vector_index.py
```

### Slow Retrieval

**Problem**: Hybrid retrieval takes > 5 seconds

**Optimizations**:
1. **Check parallel execution**: Ensure searches run concurrently
2. **Reduce top_k**: Lower `top_k` for VectorDB (30 → 20)
3. **Profile**: Use `time.time()` to identify bottleneck
4. **Cache**: Add LRU cache for common queries

---

## Performance Guidelines

### LLM Call Budgets

**Target**: < 5 LLM calls per user query

**Breakdown:**
- Preprocessing: 2 calls (parsing + validation)
- Intent classification: 1 call
- Response generation: 1-2 calls (with tool loop)

**Total**: 4-5 calls typical, 6-8 if clarification needed

### Latency Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Preprocessing | < 1s | 2 LLM calls |
| Intent classification | < 1s | 1 LLM call + agent loop |
| Hybrid retrieval | < 300ms | Parallel execution critical |
| Response generation | < 2s | 1-2 LLM calls |
| **Total** | **< 5s** | End-to-end |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| Document index (547 files) | ~100MB | In-memory JSON |
| BM25 index | ~50MB | Inverted index |
| Vector embeddings cache | ~200MB | Precomputed embeddings |
| ChromaDB | ~500MB | On-disk, memory-mapped |
| **Total** | **~850MB** | Acceptable for dev/prod |

---

## Configuration Reference

### Environment Variables

```bash
# LLM API Keys
GOOGLE_API_KEY=<your_key>              # Required
ANTHROPIC_API_KEY=<your_key>           # Optional
OPENAI_API_KEY=<your_key>              # Optional

# Agent Mode
AGENT_MODE=offline                      # offline | online

# Web Search (Online Mode)
TAVILY_API_KEY=<your_key>              # Required for online mode
WEB_SEARCH_MAX_RESULTS=5               # Max web results to retrieve

# Logging
LOG_LEVEL=INFO                          # DEBUG | INFO | WARNING | ERROR

# Fuzzy Matching
FUZZY_MATCH_THRESHOLD=80                # 0-100, higher = stricter

# Paths (optional, defaults in settings.py)
DATA_DIR=./data
CHROMA_DB_PATH=./data/output/chroma_db
```

### Model Configuration

**Current Models** (defined in `settings.py`):

```python
# Fast operations
PARSING_MODEL = "gemini-1.5-flash-002"
VALIDATION_MODEL = "gemini-1.5-flash-002"

# Complex operations
INTENT_MODEL = "gemini-1.5-flash-002"
RESPONSE_MODEL = "gemini-1.5-flash-002"

# Embeddings
EMBEDDING_MODEL = "models/embedding-001"  # Google Gemini
```

**Recommended for Production**:
- Fast ops: `claude-haiku-3-5` or `gemini-flash`
- Complex ops: `claude-sonnet-3-5` or `gemini-pro`

---

## Additional Resources

### Documentation Files

- `README.md` - User-facing guide and quickstart
- `ARCHITECTURE.md` - System architecture and design
- `RAG_IMPLEMENTATION_PRD.md` - RAG implementation details (highly detailed)
- `task.md` - Original assignment requirements
- `GRAPH_DIAGRAM.md` - Visual graph structure
- `nodes.md` - Node specifications

### External Links

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Google Gemini API](https://ai.google.dev/docs)

---

**Questions?** Check the troubleshooting section or open an issue in the repository.
