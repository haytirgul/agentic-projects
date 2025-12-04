# Product Requirements Document: Repo Analyst - Track B (Explorer Mode)

## 1. Overview

### 1.1 Purpose
Build an agentic RAG system that answers natural language questions about the httpx codebase by retrieving and synthesizing relevant code chunks with precise citations.

### 1.2 Target Codebase
- Repository: `httpx` (github.com/encode/httpx)
- Local clone (read-only access)
- Language: Python
- Size: ~15k-20k LOC

### 1.3 Key Constraints
- Must work with local repository (no GitHub API)
- Read-only operations
- Must provide file:line citations for all answers
- Answers must be concise and grounded in retrieved code

### 1.4 Quick Checklist (Project Rules)
- [ ] Are Pydantic models used for all data schemas?
- [ ] Are prompts isolated in the `prompts/` module?
- [ ] Is there a specific `logger` initialized for the module?
- [ ] Are file paths handled with `pathlib`?
- [ ] Are type hints present for all arguments and return values?
- [ ] Is `__all__` defined for public modules?

---

## 2. Goals & Success Metrics

### 2.1 Primary Goals
1. Answer natural language questions about httpx behavior
2. Provide accurate citations (file path + line ranges) for all claims
3. Demonstrate clear agentic reasoning flow
4. Maintain transparency in retrieval process

### 2.2 Success Metrics
- **Correctness**: Answers match actual httpx implementation
- **Grounding**: Every claim has valid file:line citation
- **Retrieval Quality**: Relevant code chunks in top-K results
- **Developer Experience**: One-command setup and execution
- **Answer Quality**: Concise, accurate, refuses when evidence is weak

### 2.3 Non-Goals
- Real-time code execution or testing
- Handling queries about non-httpx codebases
- Code modification or generation
- Symbol-specific queries (that's Track A)

---

## 3. Functional Requirements

### 3.1 Core Features

#### F1: Code Indexing Pipeline
**Priority**: P0
**Description**: Process httpx repository into searchable hybrid vector index with reranking

**Requirements**:
- Parse Python files into semantic chunks using AST-based chunking
- Generate dense embeddings for semantic search
- Implement hybrid search (dense + sparse retrieval)
- Include mandatory reranking with evaluation metrics
- Store in local vector database (FAISS preferred)
- Extract metadata: file path, line ranges, module context, parent structures
- Support incremental updates (nice-to-have)

**Acceptance Criteria**:
- All `.py` files in httpx repo are indexed with hybrid search capability
- Each chunk has: code content, file path, line range, dense embedding, sparse features
- Reranking evaluation metrics show improved retrieval quality
- Index build completes in <5 minutes
- Chunks preserve logical code boundaries

---

#### F2: Agentic Retrieval Flow
**Priority**: P0
**Description**: LangGraph agent with structured output parsing and fallback mechanisms

**Required Nodes**:
1. **Router**: Parses user query, plans retrieval strategy (isolated prompts)
2. **Retriever**: Executes hybrid search with reranking, returns top-K chunks
3. **Synthesizer**: Generates grounded answer with citations
4. **[Optional] Re-Retriever**: Performs follow-up searches if needed

**Agent Capabilities**:
- Initial hybrid retrieval (dense + sparse) with reranking
- Structured output parsing for all LLM responses
- LLM-powered fallback for complex queries
- Stop condition when sufficient evidence gathered
- Track retrieval history (for transparency)

**Acceptance Criteria**:
- Agent successfully answers 3 sample queries with hybrid search
- Uses structured output parsing throughout
- Can perform 1-3 retrieval iterations per query
- Logs all retrieval decisions and fallback triggers
- Handles no-result scenarios gracefully

---

#### F3: Answer Synthesis with Citations
**Priority**: P0  
**Description**: Generate concise answers with precise code references

**Requirements**:
- Extract relevant information from retrieved chunks
- Format citations as `file/path.py:line_start-line_end`
- Include multiple citations if answer spans multiple locations
- Refuse/hedge when evidence is weak or contradictory
- Keep answers concise (2-4 paragraphs max)

**Acceptance Criteria**:
- Every factual claim has citation
- Citations are valid (file exists, lines are correct)
- Answer directly addresses user question
- Shows what was searched (transparency)

---

#### F4: Conversation Memory
**Priority**: P0
**Description**: Multi-turn conversation support with context tracking

**Requirements**:
- Track conversation history across multiple queries
- Maintain context for follow-up questions
- Store conversation turns with queries and responses
- Support context continuation (e.g., "What about X?" after previous query)
- Maximum history window (e.g., last 5 turns)
- Session-based memory (in-memory, per CLI session)

**Acceptance Criteria**:
- Agent can answer follow-up questions using previous context
- Conversation history is tracked per session
- User can ask clarifying questions without repeating context
- Memory resets when CLI session ends

---

#### F5: CLI Interface
**Priority**: P0
**Description**: Interactive command-line interface for multi-turn conversations

**Requirements**:
```bash
python app.py  # Interactive mode
```
- Interactive multi-turn conversation mode
- Clear output formatting with streaming support
- Display: answer + citations + retrieval log
- Handle errors gracefully
- "Continue" prompt after each answer
- Exit command to end session

**Acceptance Criteria**:
- Works with provided sample queries
- Returns results in <30 seconds per query
- Pretty-prints answers and citations
- Shows retrieval steps taken
- Supports multi-turn conversations
- User-friendly prompts and formatting

---

### 3.2 Guardrails

#### G1: Evidence Quality Control
- Refuse to answer if no relevant chunks found
- Hedge with confidence qualifiers ("appears to", "likely", "based on X")
- Show what was searched when refusing
- Never hallucinate code that wasn't retrieved

#### G2: Read-Only Operations
- All tools/operations must be read-only
- No file modifications
- No code execution
- No external API calls (except LLM)

---

## 4. Non-Functional Requirements

### 4.1 Performance
- **Indexing**: Complete in <5 minutes for httpx repo
- **Query Response**: <30 seconds per question
- **Memory**: Run on 8GB RAM machines

### 4.2 Code Quality
- Strict typing with type hints throughout
- Google-style docstrings for all public functions/classes
- Pydantic models for all data schemas
- Prompts isolated in `prompts/` module with 8-section structure
- Centralized configuration in `settings.py` with `dotenv`/`pathlib`
- Specific logger initialized for each module
- Unit tests for chunking logic and reranking evaluation
- Integration test for end-to-end flow with hybrid search
- All public modules define `__all__`

### 4.3 Developer Experience
- One-command setup via Docker Compose (optional but recommended)
- Clear README with setup instructions
- Example queries with expected outputs
- Logging for debugging

### 4.4 Maintainability
- Modular architecture (easy to swap chunking strategies)
- Configuration file for parameters (chunk size, top-K, etc.)
- Clear separation of concerns

---

## 5. Technical Architecture

### 5.1 High-Level Components

```
┌──────────────────────────────────────────────────────┐
│    CLI Interface (app.py) - Interactive Loop         │
│    • Multi-turn conversation support                 │
│    • Streaming output                                │
│    • Continue prompts                                │
└────────────────┬─────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────┐
│         LangGraph Agent Flow                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Router  │→ │ Retriever│→ │Synthesizer│           │
│  │ (Context)│  │(Hybrid)  │  │(Citations)│           │
│  └──────────┘  └──────────┘  └──────────┘           │
│         ↓             ↓             ↓                │
│  ┌──────────────────────────────────────────┐        │
│  │    Conversation Memory Manager           │        │
│  │    • Track history (max 5 turns)         │        │
│  │    • Context injection                   │        │
│  │    • Follow-up detection                 │        │
│  └──────────────────────────────────────────┘        │
└────────────────┬─────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────┐
│         Vector Store (FAISS / ChromaDB)              │
│  - Dense embeddings (semantic search)                │
│  - Sparse features (BM25 keyword search)             │
│  - Chunk metadata (file paths, line ranges)          │
│  - Hybrid scoring + RRF reranking                    │
└──────────────────────────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────────┐
│      Code Repository (httpx)                         │
│  - Local clone (read-only)                           │
│  - AST-based chunking with metadata                  │
└──────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack

**Required**:
- Python 3.11+
- LangChain/LangGraph for agent orchestration
- FAISS for vector storage (local)
- Google gimini LLM (configurable)
- Tree-sitter or AST for parsing (optional)

**Required** (per project rules):
- Pydantic models for all data schemas
- Prompts isolated in `prompts/` module
- Hybrid search with reranking for RAG
- Centralized configuration in `settings.py` using `dotenv` and `pathlib`
- Type hints throughout codebase
- Google-style docstrings

**Recommended**:
- sentence-transformers for embeddings
- Rich for CLI formatting
- pytest for testing

### 5.3 Data Models

All data models are implemented as Pydantic models in the `models/` directory with full type hints and validation. See `models/chunk.py`, `models/retrieval.py`, and `models/agent.py` for complete implementations.

```python
# models/chunk.py
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class CodeChunk(BaseModel):
    """Pydantic model for code chunks with validation.

    Represents a chunk of code extracted from the repository with
    associated metadata and embeddings.
    """
    content: str = Field(..., description="Actual code text content")
    file_path: Path = Field(..., description="Relative path from repo root")
    start_line: int = Field(..., ge=1, description="Starting line number")
    end_line: int = Field(..., ge=1, description="Ending line number")
    chunk_type: str = Field(..., regex="^(function|class|window)$", description="Type of chunk")
    parent_context: str = Field("", description="Class/module name context")
    docstring: Optional[str] = Field(None, description="Extracted docstring")
    imports: List[str] = Field(default_factory=list, description="Relevant imports")
    embedding: List[float] = Field(default_factory=list, description="Vector representation")

    @validator('end_line')
    def end_line_must_be_after_start(cls, v, values):
        if 'start_line' in values and v < values['start_line']:
            raise ValueError('end_line must be >= start_line')
        return v

    class Config:
        arbitrary_types_allowed = True

# models/retrieval.py
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
from .chunk import CodeChunk

class RetrievalResult(BaseModel):
    """Pydantic model for retrieval results with hybrid search scores.

    Contains the results of a retrieval operation including chunks,
    similarity scores, and reranking information.
    """
    chunks: List[CodeChunk] = Field(default_factory=list, description="Retrieved code chunks")
    query: str = Field(..., description="Original search query")
    dense_scores: List[float] = Field(default_factory=list, description="Dense retrieval similarity scores")
    sparse_scores: List[float] = Field(default_factory=list, description="Sparse retrieval similarity scores")
    reranked_scores: List[float] = Field(default_factory=list, description="Reranked similarity scores")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Retrieval timestamp")

# models/conversation.py
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ConversationTurn(BaseModel):
    """Pydantic model for a single conversation turn.

    Represents one query-response pair in the conversation history.
    """
    user_query: str = Field(..., description="User's question")
    assistant_response: str = Field(..., description="Agent's answer")
    citations: List[str] = Field(default_factory=list, description="Citations used in response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Turn timestamp")
    chunks_retrieved: int = Field(0, description="Number of chunks retrieved")

class ConversationMemory(BaseModel):
    """Pydantic model for conversation memory management.

    Maintains conversation history with configurable window size.
    """
    turns: List[ConversationTurn] = Field(default_factory=list, description="Conversation history")
    max_turns: int = Field(5, description="Maximum turns to keep in memory")
    session_id: str = Field(..., description="Unique session identifier")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn and maintain max window size."""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_recent_context(self, n: int = 3) -> str:
        """Get formatted recent conversation context."""
        recent = self.turns[-n:] if len(self.turns) > n else self.turns
        return "\n\n".join([
            f"User: {t.user_query}\nAssistant: {t.assistant_response}"
            for t in recent
        ])

# models/agent.py
from typing import List, Optional
from pydantic import BaseModel, Field
from .chunk import CodeChunk
from .retrieval import RetrievalResult
from .conversation import ConversationMemory

class AgentState(BaseModel):
    """Pydantic model for LangGraph agent state with structured output.

    Maintains the state of the agent throughout the retrieval and synthesis process.
    """
    user_query: str = Field(..., description="Original user question")
    retrieval_history: List[RetrievalResult] = Field(default_factory=list, description="History of retrieval operations")
    current_chunks: List[CodeChunk] = Field(default_factory=list, description="Currently retrieved chunks")
    answer: Optional[str] = Field(None, description="Generated answer")
    citations: List[str] = Field(default_factory=list, description="File:line citations")
    should_continue: bool = Field(True, description="Whether to continue retrieval")
    conversation_memory: Optional[ConversationMemory] = Field(None, description="Conversation history")
    is_followup: bool = Field(False, description="Whether this is a follow-up question")

    class Config:
        arbitrary_types_allowed = True
```

Key models include:
- **CodeChunk**: Pydantic model for code chunks with validation and custom validators
- **RetrievalResult**: Hybrid search results with dense/sparse/reranked scores
- **ConversationTurn**: Single query-response pair with metadata
- **ConversationMemory**: Multi-turn conversation history management
- **AgentState**: LangGraph state with structured output parsing and conversation memory

All models include:
- Full type hints with Field descriptions
- Pydantic validation and custom validators
- Google-style docstrings
- Proper serialization methods
- Type-safe field constraints

---

## 6. Implementation Phases

### Phase 1: Foundation (MVP)
**Timeline**: Days 1-2

- [ ] Set up project structure
- [ ] Implement basic AST-based chunking
- [ ] Build FAISS index from httpx repo
- [ ] Create simple retrieval function (no agent yet)
- [ ] Test: Can retrieve relevant chunks for sample query

### Phase 2: Agent Flow
**Timeline**: Days 3-4

- [ ] Build LangGraph agent (Router → Retriever → Synthesizer)
- [ ] Implement citation extraction logic
- [ ] Add answer formatting
- [ ] Test: Agent answers 1 sample query end-to-end

### Phase 3: Refinement
**Timeline**: Day 5

- [ ] Add re-retrieval capability (conditional edge)
- [ ] Implement guardrails (refuse/hedge logic)
- [ ] Add retrieval transparency logging
- [ ] Test: All 3 sample queries with correct citations

### Phase 4: Polish
**Timeline**: Day 6

- [ ] CLI interface with Rich formatting
- [ ] Docker/Docker Compose setup
- [ ] Write README and ARCHITECTURE docs
- [ ] Generate example outputs
- [ ] Final testing and bug fixes

---

## 7. Sample Queries & Expected Behavior

### Query 1: "How does httpx validate SSL certificates?"
**Expected Flow**:
1. Router: Identify key terms → "SSL", "certificate", "validation"
2. Retriever: Search for SSL/TLS related code
3. Synthesizer: Explain validation logic with citations

**Expected Answer Structure**:
```
httpx validates SSL certificates through [explanation]...

Citations:
- httpx/_client.py:234-256
- httpx/_config.py:78-92
```

### Query 2: "Where in the code is proxy support implemented?"
**Expected Flow**:
1. Router: Location query → search for "proxy"
2. Retriever: Find proxy-related classes/functions
3. Synthesizer: List locations with brief description

### Query 3: "What happens if a request exceeds the configured timeout?"
**Expected Flow**:
1. Router: Behavior query → "timeout", "exceed", "error"
2. Retriever: Find timeout handling code
3. Synthesizer: Explain exception flow with citations

---

## 8. Deliverables

### 8.1 Required Files

1. **README.md**
   - Setup instructions
   - Dependencies
   - Usage examples
   - Known limitations
   - Assumptions made

2. **ARCHITECTURE.md**
   - Agent flow diagram
   - Chunking strategy explanation
   - Design decisions and rationale
   - Trade-offs considered

3. **app.py** (or main entry point)
   - CLI interface
   - Agent initialization
   - Query handling

4. **Example Outputs**
   - 3+ sample queries
   - Full answers with citations
   - Retrieval logs (optional)

5. **Docker Setup** (optional)
   - Dockerfile
   - docker-compose.yml
   - One-command deployment

### 8.2 Code Structure
```
repo-analyst/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── docker-compose.yml (optional)
├── Dockerfile (optional)
├── settings.py                # Centralized configuration with dotenv/pathlib
├── app.py                     # CLI entry point
├── src/
│   ├── indexing/
│   │   ├── chunker.py         # Code chunking logic
│   │   ├── embedder.py        # Embedding generation
│   │   └── index_builder.py   # FAISS index creation
│   ├── retrieval/
│   │   ├── vector_search.py   # Hybrid search implementation
│   │   └── reranker.py        # Mandatory reranking
│   ├── agent/
│   │   ├── graph.py           # LangGraph definition
│   │   ├── nodes/
│   │   │   ├── router.py      # Router node implementation
│   │   │   ├── retriever.py   # Retriever node implementation
│   │   │   ├── synthesizer.py # Synthesizer node implementation
│   │   │   └── conversation_memory.py  # Memory tracking node
│   │   └── state.py           # Agent state models
│   ├── synthesis/
│   │   ├── citation_extractor.py
│   │   └── answer_formatter.py
│   └── utils/
│       ├── logger.py          # Logging setup
├── models/
│   ├── __init__.py
│   ├── chunk.py               # CodeChunk and related Pydantic models
│   ├── retrieval.py           # RetrievalResult and related models
│   ├── conversation.py        # ConversationTurn and ConversationMemory models
│   └── agent.py               # AgentState and related models
├── prompts/
│   ├── __init__.py
│   ├── router_prompts.py      # Router node prompts (8-section structure)
│   ├── retriever_prompts.py   # Retriever node prompts
│   ├── synthesizer_prompts.py # Synthesizer node prompts
│   └── fallback_prompts.py    # Fuzzy -> LLM fallback prompts
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   ├── test_reranking.py       # Mandatory evaluation metrics
│   └── test_agent.py
└── data/
    ├── httpx/                 # Cloned repo (gitignored)
    └── index/                 # Generated FAISS index
```

---

## 9. Configuration Parameters

Configuration is centralized in `settings.py` using `dotenv` and `pathlib` for secure, typed configuration management.

```python
# settings.py
"""Global settings and configuration for the repo analyst project.

This module loads environment variables and provides configuration constants
that can be imported throughout the project.
"""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__all__ = [
    # Project paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "HTTX_REPO_DIR",
    "INDEX_DIR",

    # Repository settings
    "REPO_EXCLUDE_PATTERNS",

    # Chunking settings
    "CHUNK_STRATEGY",
    "MAX_CHUNK_SIZE",
    "CHUNK_OVERLAP",

    # Embedding settings
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",

    # Retrieval settings
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    "MAX_ITERATIONS",
    "VECTOR_WEIGHT",
    "SPARSE_WEIGHT",
    "RERANKING_ENABLED",

    # Conversation memory settings
    "MAX_HISTORY_TURNS",
    "ENABLE_CONVERSATION_MEMORY",

    # LLM settings
    "GOOGLE_API_KEY",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_TIMEOUT",

    # Output settings
    "LOG_LEVEL",
    "SHOW_RETRIEVAL_LOG",
    "VERBOSE_OUTPUT",
    "ENABLE_STREAMING",
]

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
HTTX_REPO_DIR = DATA_DIR / os.getenv("HTTX_REPO_DIR", "httpx")
INDEX_DIR = DATA_DIR / "index"

# ============================================================================
# REPOSITORY SETTINGS
# ============================================================================

# Patterns to exclude from indexing
REPO_EXCLUDE_PATTERNS: List[str] = [
    "tests/*",
    "docs/*",
    "*.md",
    "__pycache__/*",
    ".git/*",
    "examples/*",
    "scripts/*",
]

# ============================================================================
# CHUNKING SETTINGS
# ============================================================================

# Chunking strategy: "ast" or "window"
CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "ast")

# Maximum chunk size (lines for window, N/A for AST)
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "500"))

# Overlap between chunks (for window strategy)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding model name
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Batch size for embedding generation
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Number of top results to retrieve
TOP_K = int(os.getenv("TOP_K", "10"))

# Similarity threshold for filtering results
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# Maximum number of retrieval iterations
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))

# Hybrid search weights
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.3"))

# Enable reranking
RERANKING_ENABLED = os.getenv("RERANKING_ENABLED", "true").lower() == "true"

# ============================================================================
# CONVERSATION MEMORY SETTINGS
# ============================================================================

# Maximum conversation history turns to maintain
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "5"))

# Enable conversation memory
ENABLE_CONVERSATION_MEMORY = os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"

# ============================================================================
# LLM SETTINGS
# ============================================================================

# Google API Key (required)
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY not configured. Please set it in your .env file."
    )

# Model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# LLM parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))  # seconds

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Display options
SHOW_RETRIEVAL_LOG = os.getenv("SHOW_RETRIEVAL_LOG", "true").lower() == "true"
VERBOSE_OUTPUT = os.getenv("VERBOSE_OUTPUT", "false").lower() == "true"
ENABLE_STREAMING = os.getenv("ENABLE_STREAMING", "true").lower() == "true"

# ============================================================================
# VALIDATION
# ============================================================================

# Ensure directories exist
HTTX_REPO_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Validate chunk strategy
if CHUNK_STRATEGY not in ["ast", "window"]:
    raise ValueError(f"Invalid CHUNK_STRATEGY: {CHUNK_STRATEGY}. Must be 'ast' or 'window'")

# Validate weights sum to 1.0
if abs(VECTOR_WEIGHT + SPARSE_WEIGHT - 1.0) > 0.01:
    raise ValueError("VECTOR_WEIGHT + SPARSE_WEIGHT must equal 1.0")
```

---

## 10. Testing Strategy

### 10.1 Unit Tests
- Chunking produces valid Pydantic models with metadata
- Embeddings are generated correctly with type validation
- Citation extraction works with pathlib paths
- Hybrid retrieval returns top-K results with reranking
- All public modules define `__all__`

### 10.2 Integration Tests
- End-to-end: Query → Answer with citations
- Agent implements Fuzzy → LLM fallback correctly
- Hybrid search with reranking evaluation metrics
- Graceful failure when no results found
- Structured output parsing validation

### 10.3 Manual QA
- Test all 3 sample queries with hybrid retrieval
- Verify citations point to correct code with pathlib validation
- Check answer quality and relevance with reranking metrics
- Test edge cases (empty results, ambiguous queries, fallback triggers)

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor chunking → irrelevant retrieval | High | Test multiple chunking strategies, use AST-based |
| Slow indexing (>5 min) | Medium | Profile and optimize, use incremental indexing |
| Citations pointing to wrong lines | High | Store line numbers during chunking, validate in tests |
| Agent loops infinitely | Medium | Add max_iterations limit, stop condition |
| LLM hallucinates code | High | Strict prompting: "only use retrieved chunks" |

---

## 12. Out of Scope

- Multi-repository support
- Code execution or testing
- Real-time index updates
- Web-based UI (CLI only)
- Authentication/authorization
- Non-Python codebases
- Handling Track A queries (symbol-specific)

---

## 13. Evaluation Criteria (From Assignment)

### What They Evaluate:
1. ✅ **Correctness & Grounding**: Citations with file paths + line ranges
2. ✅ **Agent Design Clarity**: Simple and robust > complex
3. ✅ **Implementation Choices**: Rationale in ARCHITECTURE.md
4. ✅ **Code Quality**: Type hints, docs, tests
5. ✅ **Developer Experience**: One-command start

### How to Excel:
- Make chunking strategy decision explicit
- Document why you chose AST vs window
- Show retrieval transparency in output
- Clean, readable code with comments
- Comprehensive README

---

## 14. Next Steps

1. **Set up environment**: Clone httpx, install dependencies
2. **Prototype chunker**: Test AST vs window on sample files
3. **Build index**: Generate FAISS index from httpx
4. **Test retrieval**: Manually verify top-K results
5. **Implement agent**: Start with 3-node graph
6. **Iterate**: Test with sample queries, refine
7. **Document**: Write README and ARCHITECTURE
8. **Polish**: Docker, formatting, final testing

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-04  
**Owner**: Hay (AI Engineer)