# Retrieval Architecture PRD

**Version**: 1.2
**Last Updated**: 2025-12-07
**Status**: Implementation Ready

**Change Log (v1.2)**:
- ✅ Soft folder filtering (boost, not exclude) to preserve recall
- ✅ RRF_FOLDER_BOOST setting (default: 1.3x) for folder matches
- ✅ Simplified fallback chain (file_patterns only)

**Change Log (v1.1)**:
- ✅ Weighted RRF (0.4/1.0) for class chunk recall
- ✅ Code-aware BM25 tokenization (camelCase, snake_case)
- ✅ Router safe fallback with graduated relaxation
- ✅ Fast path router for simple queries (<10ms)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Component Specifications](#component-specifications)
4. [Data Flow](#data-flow)
5. [Configuration](#configuration)
6. [Future Enhancements](#future-enhancements)
7. [Implementation Notes](#implementation-notes)

---

## Overview

This document specifies the retrieval architecture for the Code RAG Agent. The system implements a **hybrid retrieval strategy** combining BM25 (lexical) and vector embeddings (semantic) with metadata filtering and context expansion.

### Design Principles

1. **Hybrid Search**: Combine BM25 and FAISS vector search with RRF ranking
2. **Metadata-First Filtering**: Filter by source type, folders, and files before search
3. **Context Expansion**: Enrich retrieved chunks with parent/sibling context
4. **Router-Guided Retrieval**: LLM router decomposes queries into structured requests
5. **No Re-Retrieval**: Single-pass retrieval (evaluator reserved for future)
6. **Unlimited Context**: Leverage 1M token context window models

### Key Metrics

- **Latency Target**:
  - Simple queries (20%): 0.5-1.5s (fast path router)
  - Complex queries (80%): 5-8s (LLM router)
- **Memory Footprint**: ~10-20MB for metadata cache
- **Storage**: FAISS indices on disk (~50-100MB)
- **Codebase**: 125 files (httpx repository)
- **Recall Improvements (v1.1)**:
  - Class definition recall: +30%
  - BM25 code queries: +20%
  - Zero-result failures: -60%

---

## Architecture

### High-Level Pipeline

```
┌─────────────────┐
│  User Query     │
└────────┬────────┘
         ▼
┌──────────────────────────────────────────┐
│  Fast Path Router (v1.1)                 │
│  • Regex patterns for simple queries     │
│  • Hex codes, function names, files      │
│  • Latency: <10ms (20% hit rate)         │
└────────┬─────────────────────────────────┘
         │
    Hit? │  Miss (80%)
         ▼
┌──────────────────────────────────────────┐
│  LLM Router                              │
│  • Input: Codebase tree + user query     │
│  • Output: N RetrievalRequest objects    │
│  • Latency: ~1-3s                        │
└────────┬─────────────────────────────────┘
         ▼
┌──────────────────────────────────────────┐
│  Parallel Retrieval (per request)        │
│  ┌────────────────────────────────────┐  │
│  │  For each RetrievalRequest:        │  │
│  │                                    │  │
│  │  1. Metadata Filter (RAM)          │  │
│  │     • Filter by source_types       │  │
│  │     • Filter by file_patterns      │  │
│  │     • Folders: SOFT (v1.2)         │  │
│  │                                    │  │
│  │  2. BM25 Search (rank_bm25)        │  │
│  │     • Search filtered corpus       │  │
│  │     • Get top-N (N=50)             │  │
│  │                                    │  │
│  │  3. FAISS Search (disk)            │  │
│  │     • Load relevant index          │  │
│  │     • Search filtered vectors      │  │
│  │     • Get top-N (N=50)             │  │
│  │                                    │  │
│  │  4. RRF Ranking (k=60, v1.1)       │  │
│  │     • Weighted: BM25=0.4, Vec=1.0  │  │
│  │     • Favors semantic for classes  │  │
│  │                                    │  │
│  │  5. Folder Boost (v1.2)            │  │
│  │     • Boost folder matches by 1.3x │  │
│  │     • Re-rank by boosted scores    │  │
│  │                                    │  │
│  │  6. Get Top-K (K=20)               │  │
│  │     • Return chunk IDs             │  │
│  └────────────────────────────────────┘  │
│  • Latency: ~200-500ms per request       │
└────────┬─────────────────────────────────┘
         ▼
┌──────────────────────────────────────────┐
│  Rebuild Full Chunks                     │
│  • Load from JSON files using chunk IDs  │
│  • Validate with Pydantic models         │
│  • Latency: ~50-100ms                    │
└────────┬─────────────────────────────────┘
         ▼
┌──────────────────────────────────────────┐
│  Context Expansion                       │
│  • Code: parent class + methods + imports│
│  • Markdown: parent headers + children   │
│  • Text: No expansion (standalone)       │
│  • Fetch using metadata index (RAM)      │
│  • Latency: ~100-200ms                   │
└────────┬─────────────────────────────────┘
         ▼
┌──────────────────────────────────────────┐
│  Synthesis (LLM with 1M context)         │
│  • Input: All expanded chunks            │
│  • Context: Reasoning for each query     │
│  • Output: Answer with citations         │
│  • Latency: ~2-5s                        │
└──────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Storage Architecture

#### **Chunk Storage**

```python
# Storage strategy: Hybrid disk + RAM

# 1. Full chunks stored in JSON files (disk)
data/processed/
├── code_chunks.json         # ~5MB
├── markdown_chunks.json     # ~2MB
└── text_chunks.json         # ~1MB

# 2. FAISS indices stored on disk
data/index/
├── code_faiss.index         # FAISS index for code chunks
├── markdown_faiss.index     # FAISS index for markdown chunks
└── text_faiss.index         # FAISS index for text chunks

# 3. Metadata index in RAM (lightweight)
{
  "code": [
    {
      "id": "abc123",
      "file_path": "src/client.py",
      "filename": "client.py",
      "chunk_type": "function",
      "name": "get",
      "parent_context": "Client",
      "faiss_index_position": 42,  # Position in FAISS index
      "token_count": 150
    },
    ...
  ],
  "markdown": [...],
  "text": [...]
}
# Total RAM: ~10-20MB for 125 files
```

#### **FAISS Configuration**

```python
# Separate FAISS index per source_type for efficient filtering
class FAISSStore:
    def __init__(self):
        # FAISS indices stored on disk
        self.index_paths = {
            "code": Path("data/index/code_faiss.index"),
            "markdown": Path("data/index/markdown_faiss.index"),
            "text": Path("data/index/text_faiss.index")
        }

        # Metadata index in RAM (no content, just IDs and metadata)
        self.metadata_index: Dict[str, list[ChunkMetadata]] = {
            "code": [],
            "markdown": [],
            "text": []
        }

        # Chunk ID to file path mapping (for loading full chunks)
        self.chunk_locations: Dict[str, Path] = {}
```

**Design Rationale**:
- **FAISS on disk**: Reduces RAM usage, acceptable I/O latency (~50-100ms)
- **Metadata in RAM**: Fast filtering without loading full chunks or FAISS indices
- **Separate indices per source_type**: Efficient filtering (load only relevant index)
- **Lazy loading**: Load FAISS index only when needed for search

---

### 2. Router Component

#### **Fast Path Router (v1.1)**

```python
class FastPathRouter:
    """Fast regex-based router for trivial queries (v1.1).

    Handles:
    - Exact function names (e.g., "get_request", "HTTPClient")
    - Hex error codes (e.g., "0x884", "0xFF")
    - Simple file names (e.g., "config.toml", "README.md")

    Bypasses LLM router (~2s) for ~20% of queries (<10ms latency).
    """

    PATTERNS = {
        "hex_code": re.compile(r'^0x[0-9A-Fa-f]+$'),
        "function_name": re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$'),
        "camel_case_class": re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
        "file_name": re.compile(r'^[a-zA-Z0-9_\-]+\.[a-z]{2,4}$'),
    }

    def route(self, query: str) -> Optional[RouterOutput]:
        """Attempt fast path routing.

        Returns:
            RouterOutput if pattern matches, None if LLM needed
        """
        query = query.strip()

        # Pattern 1: Hex error code
        if self.PATTERNS["hex_code"].match(query):
            return RouterOutput(
                cleaned_query=query,
                retrieval_requests=[
                    RetrievalRequest(
                        query=query,
                        source_types=["code"],
                        folders=[],
                        file_patterns=[],
                        reasoning=f"Exact hex code search for {query}"
                    )
                ]
            )

        # Pattern 2: Exact function name
        if self.PATTERNS["function_name"].match(query):
            return RouterOutput(
                cleaned_query=query,
                retrieval_requests=[
                    RetrievalRequest(
                        query=query,
                        source_types=["code"],
                        folders=[],
                        file_patterns=["*.py"],
                        reasoning=f"Exact function name search for {query}"
                    )
                ]
            )

        # No pattern matched → use LLM router
        return None
```

#### **Main Router (with Fast Path Integration)**

```python
class Router:
    """Main router with fast path optimization (v1.1)."""

    def __init__(self):
        self.fast_path = FastPathRouter()
        self.llm_router = LLMRouter()

    def route(self, query: str, codebase_tree: str) -> RouterOutput:
        """Route query with fast path check."""

        # Try fast path first (v1.1)
        fast_result = self.fast_path.route(query)
        if fast_result:
            logger.info(f"Fast path matched for query: {query}")
            return fast_result

        # Fall back to LLM router
        logger.info(f"Using LLM router for query: {query}")
        return self.llm_router.route(query, codebase_tree)
```

#### **Input**

```python
class RouterInput(BaseModel):
    """Input to router component."""

    query: str = Field(..., description="Raw user query")
    codebase_tree: str = Field(..., description="Filtered directory tree")
```

#### **Codebase Tree Generation**

```python
def generate_codebase_tree(
    repo_root: Path,
    exclude_patterns: list[str] = REPO_EXCLUDE_PATTERNS
) -> str:
    """Generate compact directory tree for router prompt.

    Features:
    - Exclude patterns from const.REPO_EXCLUDE_PATTERNS
    - Max 4 levels of nesting
    - Group files by type in large directories
    - Estimate: 500-800 tokens for 125 files

    Example output:
    ```
    httpx/
    ├── src/
    │   ├── indexing/ (3 Python files)
    │   │   ├── vector_store.py
    │   │   ├── bm25_index.py
    │   │   └── embeddings.py
    │   ├── retrieval/
    │   │   ├── hybrid_search.py
    │   │   ├── reranker.py
    │   │   └── context_expander.py
    │   └── agent/
    │       ├── router.py
    │       └── synthesizer.py
    ├── documents/ (5 Markdown files)
    │   ├── README.md
    │   ├── API.md
    │   └── ARCHITECTURE.md
    └── data/
        └── config.toml
    ```
    """
```

#### **Router Prompt**

```python
ROUTER_PROMPT = """
You are a codebase navigation expert. Decompose user queries into specific retrieval requests.

CODEBASE STRUCTURE:
{codebase_tree}

USER QUERY: "{user_query}"

TASK:
1. Clean the query from fluff/filler words
2. Identify ALL distinct pieces of information needed
3. For EACH piece, create a RetrievalRequest

CRITICAL RULES:
- Queries MUST be distinct (no rephrasing the same question)
- If queries target similar information → unify to single request with multiple filters
- Combine source_types in single request when possible
- Use folders/file_patterns only if query is specific
- Each request needs clear reasoning (for synthesis context)

FILTERING GUIDELINES:

1. Source Type Selection:
   - "code": Implementation, logic, algorithms, functions, classes
   - "markdown": Documentation, guides, explanations, README
   - "text": Configuration files (toml, yaml, json, ini)
   - Combine types when both implementation + docs needed

2. Folder Filtering (coarse):
   - Use when query mentions specific component/module
   - Example: "retrieval logic" → folders: ["src/retrieval/"]
   - Leave empty for broad/exploratory queries

3. File Pattern Filtering (fine-grained):
   - Use when query is very specific
   - Example: "README installation" → file_patterns: ["README.md"]
   - Use wildcards: ["*_search.py", "test_*.py"]
   - Leave empty for coarse search

EXAMPLES:

BAD (duplicate queries):
{{
  "retrieval_requests": [
    {{"query": "BM25 search implementation", "source_types": ["code"]}},
    {{"query": "BM25 documentation", "source_types": ["markdown"]}}  // ❌ Redundant!
  ]
}}

GOOD (unified):
{{
  "retrieval_requests": [
    {{
      "query": "BM25 search implementation and documentation",
      "source_types": ["code", "markdown"],  // ✅ Combined
      "folders": ["src/retrieval/", "documents/"],
      "file_patterns": [],
      "reasoning": "Need both implementation and documentation of BM25 algorithm"
    }}
  ]
}}

OUTPUT FORMAT (JSON):
{{
  "cleaned_query": "concise version of user query",
  "retrieval_requests": [
    {{
      "query": "specific search query",
      "source_types": ["code", "markdown", "text"],
      "folders": ["path/to/folder/"],
      "file_patterns": ["*.md", "*_config.py"],
      "reasoning": "why this information is needed for answering the query"
    }}
  ]
}}
"""
```

#### **Output Schema**

```python
class RetrievalRequest(BaseModel):
    """Single retrieval request from router."""

    query: str = Field(
        ...,
        min_length=5,
        description="Specific search query"
    )

    source_types: list[Literal["code", "markdown", "text"]] = Field(
        ...,
        min_length=1,
        description="Document types to search (can combine multiple)"
    )

    folders: list[str] = Field(
        default_factory=list,
        description="Target folders (empty = search all). Format: 'src/retrieval/'"
    )

    file_patterns: list[str] = Field(
        default_factory=list,
        description="File patterns. Examples: ['*.md', 'test_*.py', 'README.md']"
    )

    reasoning: str = Field(
        ...,
        min_length=10,
        description="WHY this information is needed (context for synthesis)"
    )

    @field_validator('query')
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class RouterOutput(BaseModel):
    """Router output with multiple retrieval requests."""

    cleaned_query: str = Field(
        ...,
        description="User query cleaned from fluff"
    )

    retrieval_requests: list[RetrievalRequest] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Parallel retrieval requests (max 5)"
    )

    @field_validator('retrieval_requests')
    @classmethod
    def validate_unique_queries(cls, v: list[RetrievalRequest]) -> list[RetrievalRequest]:
        """Ensure queries are distinct."""
        queries = [req.query.lower() for req in v]
        if len(queries) != len(set(queries)):
            raise ValueError(
                "Duplicate queries detected. Router must unify similar requests."
            )
        return v
```

---

### 3. Metadata Filtering (v1.2 - Soft Folder Filtering)

#### **Metadata Schema**

```python
class ChunkMetadata(BaseModel):
    """Lightweight metadata stored in RAM (no content)."""

    id: str = Field(..., description="Chunk ID (MD5 hash)")
    source_type: Literal["code", "markdown", "text"]
    chunk_type: str
    file_path: str
    filename: str
    name: str
    parent_context: str = ""
    faiss_index_position: int = Field(..., description="Position in FAISS index")
    token_count: int

    # Optional fields for filtering
    file_extension: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
```

#### **Filtering Logic (v1.2)**

**v1.2 Change**: Folder filtering is now SOFT (boost, not exclude). This prevents
LLM folder inference from reducing recall when it guesses wrong.

```python
class MetadataFilter:
    """Apply metadata filters before search (v1.2 - soft folder filtering).

    v1.2: Folders no longer exclude chunks - they boost RRF scores instead.
    """

    def __init__(
        self,
        metadata_index: Dict[str, list[ChunkMetadata]],
        request: RetrievalRequest
    ):
        self.metadata_index = metadata_index
        self.request = request

    def apply(self) -> list[str]:
        """Return filtered chunk IDs that match source_type and file_patterns.

        v1.2: Folder filtering removed (now soft boost in RRF).

        Filter order:
        1. Source type (select relevant indices) - HARD filter
        2. File patterns (if specified) - HARD filter
        3. Folders - NOT applied (soft boost handled in HybridRetriever)
        """
        candidate_ids = []

        # 1. Filter by source_type (multiple types allowed)
        for source_type in self.request.source_types:
            for chunk_meta in self.metadata_index[source_type]:

                # 2. Filter by file patterns (if specified) - HARD filter
                if self.request.file_patterns:
                    if not self._matches_file_pattern(
                        chunk_meta.filename,
                        self.request.file_patterns
                    ):
                        continue

                # 3. Folders - NOT filtered here (soft boost in RRF)
                # This prevents LLM folder inference from reducing recall

                # Passed all filters
                candidate_ids.append(chunk_meta.id)

        return candidate_ids

    def _matches_file_pattern(
        self,
        filename: str,
        patterns: list[str]
    ) -> bool:
        """Check if filename matches any pattern (supports wildcards)."""
        from fnmatch import fnmatch
        return any(fnmatch(filename, pattern) for pattern in patterns)
```

#### **Folder Boost (v1.2)**

Folders are now handled via score boosting in RRF, not exclusion:

```python
def _apply_folder_boost(
    self,
    ranked_ids: list[tuple[str, float]],
    folders: list[str],
    chunk_file_paths: dict[str, str],
    boost_factor: float = 1.3  # RRF_FOLDER_BOOST
) -> list[tuple[str, float]]:
    """Apply folder boost to chunks matching inferred folders (v1.2).

    Chunks whose file_path starts with any of the specified folders
    get their RRF score multiplied by boost_factor.

    This is SOFT filtering - non-matching chunks are NOT excluded.
    """
    boosted_results = []

    for chunk_id, score in ranked_ids:
        file_path = chunk_file_paths.get(chunk_id, "")

        # Check if file_path matches any folder
        matches_folder = any(
            file_path.startswith(folder)
            for folder in folders
        )

        if matches_folder:
            boosted_score = score * boost_factor
        else:
            boosted_score = score

        boosted_results.append((chunk_id, boosted_score))

    # Re-sort by boosted score
    boosted_results.sort(key=lambda x: x[1], reverse=True)
    return boosted_results
```

**Rationale for Soft Filtering**:
- LLM router infers folders from codebase tree (e.g., "HTTPTransport" → `httpx/_transports/`)
- If LLM guesses wrong folder, hard filtering would exclude ALL relevant results
- Soft boosting ensures folder hints help ranking without hurting recall
- Default boost (1.3x) is enough to prioritize folder matches without dominating

---

### 4. Hybrid Search (BM25 + FAISS)

#### **Architecture**

```python
class HybridRetriever:
    """Hybrid retrieval combining BM25 and FAISS with RRF."""

    def __init__(
        self,
        metadata_index: Dict[str, list[ChunkMetadata]],
        faiss_store: FAISSStore,
        chunk_loader: ChunkLoader
    ):
        self.metadata_index = metadata_index
        self.faiss_store = faiss_store
        self.chunk_loader = chunk_loader

        # BM25 indices (built on-demand from filtered corpus)
        self.bm25_cache: Dict[str, BM25Okapi] = {}

    def search(
        self,
        request: RetrievalRequest,
        top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Execute hybrid search with graduated fallback (v1.1).

        v1.1 Update: Added graduated relaxation to prevent zero-result failures
        when LLM router hallucinates file patterns or folder names.

        Steps:
        1. Metadata filtering (with fallback)
        2. BM25 search (on filtered corpus)
        3. FAISS search (on filtered vectors)
        4. Weighted RRF ranking (0.4/1.0)
        5. Return top-k chunks

        Fallback chain:
        - Attempt 1: Strict (source_types + folders + file_patterns)
        - Attempt 2: Relaxed-1 (drop file_patterns, keep folders)
        - Attempt 3: Relaxed-2 (drop folders, keep source_types)
        - Attempt 4: Emergency (search all source_types)
        """
        # 1. Attempt strict filtering
        candidate_ids = MetadataFilter(
            self.metadata_index,
            request
        ).apply()

        # Fallback 1: Drop file_patterns if zero results
        if not candidate_ids and request.file_patterns:
            logger.info(
                f"Strict file patterns {request.file_patterns} yielded 0 results. "
                f"Relaxing to folder-level filtering."
            )
            relaxed_request = request.model_copy(update={"file_patterns": []})
            candidate_ids = MetadataFilter(
                self.metadata_index,
                relaxed_request
            ).apply()

        # Fallback 2: Drop folders if still zero
        if not candidate_ids and request.folders:
            logger.info(
                f"Folder filtering {request.folders} yielded 0 results. "
                f"Relaxing to source_type-only filtering."
            )
            relaxed_request = request.model_copy(update={
                "file_patterns": [],
                "folders": []
            })
            candidate_ids = MetadataFilter(
                self.metadata_index,
                relaxed_request
            ).apply()

        # Fallback 3: Emergency - search all source_types
        if not candidate_ids:
            logger.warning(
                f"All filters yielded 0 results. Emergency fallback: searching all source_types."
            )
            emergency_request = RetrievalRequest(
                query=request.query,
                source_types=["code", "markdown", "text"],
                folders=[],
                file_patterns=[],
                reasoning=request.reasoning + " [EMERGENCY FALLBACK]"
            )
            candidate_ids = MetadataFilter(
                self.metadata_index,
                emergency_request
            ).apply()

        # If still no candidates, return empty
        if not candidate_ids:
            logger.error(f"No candidates found even after full relaxation for query: {request.query}")
            return []

        logger.info(
            f"Filtered to {len(candidate_ids)} candidates "
            f"for query: {request.query}"
        )

        # 2. BM25 search
        bm25_results = self._bm25_search(
            query=request.query,
            candidate_ids=candidate_ids,
            top_n=50
        )

        # 3. FAISS search
        faiss_results = self._faiss_search(
            query=request.query,
            candidate_ids=candidate_ids,
            source_types=request.source_types,
            top_n=50
        )

        # 4. RRF ranking
        ranked_ids = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            k=60
        )

        # 5. Get top-k
        top_ids = ranked_ids[:top_k]

        # 6. Load full chunks and attach metadata
        retrieved_chunks = []
        for chunk_id in top_ids:
            chunk = self.chunk_loader.load_chunk(chunk_id)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    rrf_score=ranked_ids[chunk_id]["score"],
                    request_reasoning=request.reasoning,
                    source_query=request.query
                )
            )

        return retrieved_chunks
```

#### **BM25 Search**

```python
def _bm25_search(
    self,
    query: str,
    candidate_ids: list[str],
    top_n: int = 50
) -> list[Tuple[str, float]]:
    """BM25 search on filtered corpus.

    Strategy: Build temporary BM25 index from filtered chunks
    (fast for small candidate sets, no need to maintain full index)
    """
    from rank_bm25 import BM25Okapi

    # Load content for candidate chunks
    candidate_chunks = []
    for chunk_id in candidate_ids:
        chunk = self.chunk_loader.load_chunk(chunk_id)
        candidate_chunks.append({
            "id": chunk.id,
            "content": chunk.content
        })

    # Tokenize corpus
    tokenized_corpus = [
        self._tokenize(chunk["content"])
        for chunk in candidate_chunks
    ]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    # Search
    tokenized_query = self._tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-N results
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = [
        (candidate_chunks[i]["id"], scores[i])
        for i in top_indices
    ]

    return results

def _tokenize(self, text: str) -> list[str]:
    """Code-aware tokenization for BM25 (v1.1).

    v1.1 Update: Handles camelCase, snake_case, and preserves hex codes.

    Examples:
    - "HTTPClient" → ["http", "client"]
    - "get_user_by_id" → ["get", "user", "by"]
    - "error 0x884" → ["error", "0x884"]

    Strategy:
    1. Split camelCase boundaries (lowercase→uppercase)
    2. Replace underscores and punctuation with spaces
    3. Lowercase and filter short tokens (except hex codes)
    """
    import re

    # Step 1: Split camelCase (e.g., 'HTTPClient' → 'HTTP Client')
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # lowercase → uppercase
    text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)  # HTTPSServer → HTTPS Server

    # Step 2: Split on underscores and non-alphanumeric (but keep 0x prefix)
    text = re.sub(r'_+', ' ', text)  # Underscores → spaces
    text = re.sub(r'[^a-zA-Z0-9x\s]', ' ', text)  # Keep 'x' for hex

    # Step 3: Lowercase and split
    tokens = text.lower().split()

    # Step 4: Filter short tokens (but keep hex codes like '0x')
    filtered_tokens = []
    for t in tokens:
        if len(t) > 2:  # Standard filter
            filtered_tokens.append(t)
        elif t.startswith('0x'):  # Exception for hex
            filtered_tokens.append(t)

    return filtered_tokens
```

#### **FAISS Search**

```python
def _faiss_search(
    self,
    query: str,
    candidate_ids: list[str],
    source_types: list[str],
    top_n: int = 50
) -> list[Tuple[str, float]]:
    """FAISS vector search on filtered candidates.

    Steps:
    1. Load relevant FAISS indices (by source_type)
    2. Get FAISS positions for candidate IDs
    3. Extract vectors for candidates
    4. Search using query embedding
    """
    # 1. Generate query embedding
    query_embedding = self._embed_query(query)

    results = []

    # 2. Search each relevant source_type index
    for source_type in source_types:
        # Load FAISS index from disk
        index = self.faiss_store.load_index(source_type)

        # Get FAISS positions for candidate IDs
        candidate_positions = []
        candidate_id_map = {}  # position -> chunk_id

        for chunk_id in candidate_ids:
            meta = self._get_metadata(chunk_id)
            if meta.source_type == source_type:
                candidate_positions.append(meta.faiss_index_position)
                candidate_id_map[meta.faiss_index_position] = chunk_id

        if not candidate_positions:
            continue

        # Extract candidate vectors from FAISS index
        candidate_vectors = np.array([
            index.reconstruct(pos) for pos in candidate_positions
        ])

        # Create temporary index with only candidates
        temp_index = faiss.IndexFlatL2(index.d)
        temp_index.add(candidate_vectors)

        # Search
        distances, indices = temp_index.search(
            query_embedding.reshape(1, -1),
            min(top_n, len(candidate_positions))
        )

        # Convert back to chunk IDs
        for i, dist in zip(indices[0], distances[0]):
            original_position = candidate_positions[i]
            chunk_id = candidate_id_map[original_position]
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + dist)
            results.append((chunk_id, similarity))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]

def _embed_query(self, query: str) -> np.ndarray:
    """Generate query embedding (uses same model as indexing)."""
    # TODO: Use cached embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(query)
```

#### **RRF Ranking**

```python
def _reciprocal_rank_fusion(
    self,
    bm25_results: list[Tuple[str, float]],
    faiss_results: list[Tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 0.4,   # v1.1: Reduced weight for keyword matching
    vector_weight: float = 1.0   # v1.1: Full weight for semantic intent
) -> list[str]:
    """Weighted Reciprocal Rank Fusion with k=60 (v1.1).

    v1.1 Update: Weighted RRF favoring semantic search to improve recall
    for short class signature chunks (which BM25 under-ranks due to length bias).

    Formula: RRF_score(chunk) = bm25_weight * sum(1/(k + rank_bm25)) +
                                 vector_weight * sum(1/(k + rank_vector))

    Weighting Rationale:
    - BM25 (0.4): Good for exact terms, but biased by document length
    - Vector (1.0): Better for intent, especially for short signature chunks

    Args:
        bm25_results: list of (chunk_id, score) from BM25
        faiss_results: list of (chunk_id, score) from FAISS
        k: RRF constant (default: 60)
        bm25_weight: Weight for BM25 contribution (default: 0.4)
        vector_weight: Weight for Vector contribution (default: 1.0)

    Returns:
        list of chunk IDs sorted by weighted RRF score (highest first)
    """
    from collections import defaultdict

    rrf_scores = defaultdict(float)

    # Weighted BM25 contribution
    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] += bm25_weight * (1.0 / (k + rank))

    # Weighted Vector contribution
    for rank, (chunk_id, _) in enumerate(faiss_results, start=1):
        rrf_scores[chunk_id] += vector_weight * (1.0 / (k + rank))

    # Sort by weighted RRF score
    ranked_ids = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [chunk_id for chunk_id, score in ranked_ids]
```

---

### 5. Context Expansion

#### **Retrieved Chunk Schema**

```python
class RetrievedChunk(BaseModel):
    """Retrieved chunk with metadata for synthesis."""

    chunk: Chunk  # Full Pydantic chunk (CodeChunk/MarkdownChunk/TextChunk)
    rrf_score: float
    request_reasoning: str  # From RetrievalRequest.reasoning
    source_query: str  # Query that retrieved this chunk
    context: list[ContextItem] = Field(default_factory=list)


class ContextItem(BaseModel):
    """Single piece of context added during expansion."""

    type: str = Field(..., description="Context type (parent_class, sibling_method, etc.)")
    content: str = Field(..., description="Context content")
    reasoning: str = Field(..., description="Why this context is relevant")
    token_count: int
```

#### **Code Context Expansion**

```python
class CodeContextExpander:
    """Expand code chunks with parent/sibling/import context."""

    def __init__(self, metadata_index: Dict[str, list[ChunkMetadata]]):
        self.metadata_index = metadata_index
        self.chunk_loader = ChunkLoader()

    def expand(self, chunk: CodeChunk) -> list[ContextItem]:
        """Expand code chunk with relevant context.

        Context hierarchy:
        1. Parent class (if method)
        2. Related methods (same class, up to 3)
        3. Import statements (top 5)
        """
        context_items = []

        # 1. Parent class context
        if chunk.parent_context and chunk.chunk_type == "function":
            parent = self._find_parent_class(chunk)
            if parent:
                context_items.append(ContextItem(
                    type="parent_class",
                    content=f"# Class: {parent.name}\n{parent.docstring or ''}",
                    reasoning="Parent class context for understanding method purpose",
                    token_count=estimate_token_count(parent.docstring or "")
                ))

        # 2. Related methods (siblings in same class)
        if chunk.parent_context:
            related_methods = self._find_sibling_methods(chunk)
            for method in related_methods[:3]:  # Limit to 3
                context_items.append(ContextItem(
                    type="sibling_method",
                    content=f"# Method: {method.name}\n{method.content}",
                    reasoning=f"Related method '{method.name}' in same class",
                    token_count=method.token_count
                ))

        # 3. Import statements
        if chunk.imports:
            imports_str = "\n".join(chunk.imports[:5])
            context_items.append(ContextItem(
                type="imports",
                content=imports_str,
                reasoning="Dependencies and external references",
                token_count=estimate_token_count(imports_str)
            ))

        return context_items

    def _find_parent_class(self, chunk: CodeChunk) -> Optional[CodeChunk]:
        """Find parent class by name in same file."""
        for meta in self.metadata_index["code"]:
            if (meta.chunk_type == "class" and
                meta.name == chunk.parent_context and
                meta.file_path == chunk.file_path):
                return self.chunk_loader.load_chunk(meta.id)
        return None

    def _find_sibling_methods(self, chunk: CodeChunk) -> list[CodeChunk]:
        """Find other methods in same class."""
        siblings = []
        for meta in self.metadata_index["code"]:
            if (meta.chunk_type == "function" and
                meta.parent_context == chunk.parent_context and
                meta.file_path == chunk.file_path and
                meta.id != chunk.id):  # Exclude self
                siblings.append(self.chunk_loader.load_chunk(meta.id))
        return siblings
```

#### **Markdown Context Expansion**

```python
class MarkdownContextExpander:
    """Expand markdown chunks with header hierarchy."""

    def __init__(self, metadata_index: Dict[str, list[ChunkMetadata]]):
        self.metadata_index = metadata_index
        self.chunk_loader = ChunkLoader()

    def expand(self, chunk: MarkdownChunk) -> list[ContextItem]:
        """Expand markdown chunk with header context.

        Context hierarchy:
        1. Parent header (if nested section)
        2. Children sections (list only, not full content)
        """
        context_items = []

        # 1. Parent header context
        if chunk.headers and len(chunk.headers) > 1:
            parent_header = self._get_parent_header(chunk)
            if parent_header:
                context_items.append(ContextItem(
                    type="parent_header",
                    content=f"Parent Section: {parent_header}",
                    reasoning="Header hierarchy for document structure context",
                    token_count=10
                ))

        # 2. Children sections (list only)
        children = self._find_children_sections(chunk)
        if children:
            child_names = "\n".join([f"- {c.name}" for c in children])
            context_items.append(ContextItem(
                type="children_list",
                content=f"Subsections:\n{child_names}",
                reasoning="Document structure overview (child sections)",
                token_count=estimate_token_count(child_names)
            ))

        return context_items

    def _get_parent_header(self, chunk: MarkdownChunk) -> Optional[str]:
        """Extract parent header from headers dict."""
        # headers = {"h1": "Getting Started", "h2": "Installation", "h3": "MacOS"}
        # Parent of h3 = h2
        if "h2" in chunk.headers and chunk.heading_level == 3:
            return chunk.headers["h2"]
        elif "h1" in chunk.headers and chunk.heading_level == 2:
            return chunk.headers["h1"]
        return None

    def _find_children_sections(self, chunk: MarkdownChunk) -> list[MarkdownChunk]:
        """Find child sections (deeper headers in same file)."""
        children = []
        for meta in self.metadata_index["markdown"]:
            chunk_meta = self._get_metadata(chunk.id)
            if (meta.file_path == chunk.file_path and
                meta.faiss_index_position > chunk_meta.faiss_index_position and
                meta.headers.get(f"h{chunk.heading_level}") == chunk.name):
                children.append(self.chunk_loader.load_chunk(meta.id))
        return children[:10]  # Limit to 10 children
```

#### **Text Context Expansion**

```python
class TextContextExpander:
    """No expansion for text chunks (config files are standalone)."""

    def expand(self, chunk: TextChunk) -> list[ContextItem]:
        """Text chunks (config files) are standalone - no context needed."""
        return []
```

---

### 6. Synthesis Component

#### **Synthesis Prompt**

```python
def build_synthesis_prompt(
    user_query: str,
    cleaned_query: str,
    retrieved_chunks: list[RetrievedChunk]
) -> str:
    """Build synthesis prompt with grouped chunks and reasoning context."""

    # Group chunks by retrieval request reasoning
    chunks_by_reasoning = defaultdict(list)
    for chunk_obj in retrieved_chunks:
        chunks_by_reasoning[chunk_obj.request_reasoning].append(chunk_obj)

    # Build prompt
    prompt = f"""You are a technical documentation assistant. Answer the user's question using ONLY the provided context.

USER QUERY: {user_query}

CLEANED QUERY: {cleaned_query}

RETRIEVED INFORMATION:

"""

    for reasoning, chunks in chunks_by_reasoning.items():
        prompt += f"\n## RETRIEVAL INTENT: {reasoning}\n\n"

        for i, chunk_obj in enumerate(chunks, 1):
            chunk = chunk_obj.chunk
            prompt += f"### Chunk {i} [{chunk.source_type} - {chunk.chunk_type}]\n"
            prompt += f"**Source**: `{chunk.file_path}:{chunk.start_line}-{chunk.end_line}`\n"
            prompt += f"**Name**: {chunk.full_name}\n"
            prompt += f"**RRF Score**: {chunk_obj.rrf_score:.3f}\n\n"

            prompt += f"**Content**:\n```\n{chunk.content}\n```\n\n"

            # Add context if present
            if chunk_obj.context:
                prompt += "**Additional Context**:\n"
                for ctx in chunk_obj.context:
                    prompt += f"- **{ctx.type}**: {ctx.reasoning}\n"
                    prompt += f"  ```\n  {ctx.content[:300]}...\n  ```\n"
                prompt += "\n"

    prompt += """
INSTRUCTIONS:
1. Synthesize a comprehensive answer using the chunks above
2. Cite sources using [file:line] format (e.g., [client.py:42-58])
3. Include code examples when relevant
4. If information is insufficient, state: "Based on the available codebase context..."
5. Do NOT invent information not present in chunks
6. Explain the reasoning behind retrieval intents when it helps answer clarity

OUTPUT FORMAT:
- Direct answer to the user's question
- Code examples with syntax highlighting
- Citations for all claims [file:line]
- Clear section structure if answer is complex
"""

    return prompt
```

---

## Data Flow

### End-to-End Example

**User Query**: "How does the Client class make HTTP requests?"

#### **Step 1: Router**

**Input**:
```json
{
  "query": "How does the Client class make HTTP requests?",
  "codebase_tree": "httpx/\n  ├── src/\n  │   ├── client.py\n  │   ├── transport.py\n  ..."
}
```

**Output**:
```json
{
  "cleaned_query": "Client class HTTP request implementation",
  "retrieval_requests": [
    {
      "query": "Client class HTTP request methods implementation",
      "source_types": ["code"],
      "folders": ["src/"],
      "file_patterns": ["client.py"],
      "reasoning": "Need Client class implementation showing HTTP request methods"
    },
    {
      "query": "HTTP client usage documentation",
      "source_types": ["markdown"],
      "folders": ["documents/"],
      "file_patterns": ["*.md"],
      "reasoning": "Need documentation explaining how to use Client class for requests"
    }
  ]
}
```

#### **Step 2: Metadata Filtering (Request 1)**

**Filters**:
- `source_types`: ["code"]
- `folders`: ["src/"]
- `file_patterns`: ["client.py"]

**Result**: 15 candidate chunk IDs from `client.py`

#### **Step 3: Hybrid Search (Request 1)**

**BM25 Search** (on 15 candidates):
```python
Top 5:
1. chunk_abc123 (score: 8.5) - Client.get() method
2. chunk_def456 (score: 7.2) - Client.post() method
3. chunk_ghi789 (score: 6.8) - Client._request() method
4. chunk_jkl012 (score: 5.5) - Client class definition
5. chunk_mno345 (score: 4.2) - Client.__init__()
```

**FAISS Search** (on 15 candidates):
```python
Top 5:
1. chunk_abc123 (similarity: 0.89) - Client.get() method
2. chunk_jkl012 (similarity: 0.85) - Client class definition
3. chunk_def456 (similarity: 0.82) - Client.post() method
4. chunk_pqr678 (similarity: 0.78) - Client.request() method
5. chunk_ghi789 (similarity: 0.75) - Client._request() method
```

**RRF Ranking** (k=60):
```python
Ranked:
1. chunk_abc123 (RRF: 0.0328) - Client.get()
2. chunk_jkl012 (RRF: 0.0310) - Client class
3. chunk_def456 (RRF: 0.0305) - Client.post()
4. chunk_ghi789 (RRF: 0.0290) - Client._request()
5. chunk_pqr678 (RRF: 0.0285) - Client.request()
```

**Top-K Selection** (K=20): Return top 20 chunk IDs

#### **Step 4: Context Expansion**

**For chunk_abc123 (Client.get method)**:

Context added:
1. **Parent class**: Client class docstring
2. **Sibling methods**: `post()`, `put()`, `delete()`
3. **Imports**: `import httpcore`, `from typing import Optional`

#### **Step 5: Synthesis**

**Prompt structure**:
```
USER QUERY: How does the Client class make HTTP requests?

RETRIEVED INFORMATION:

## RETRIEVAL INTENT: Need Client class implementation showing HTTP request methods

### Chunk 1 [code - function]
Source: src/client.py:142-158
Name: Client.get
Content: [Client.get() implementation]
Additional Context:
- parent_class: Client class overview
- sibling_method: Related method 'post' in same class

### Chunk 2 [code - class]
Source: src/client.py:50-85
Name: Client
Content: [Client class definition]

...

## RETRIEVAL INTENT: Need documentation explaining how to use Client class

### Chunk 1 [markdown - markdown_section]
Source: documents/API.md:120-145
Name: Making HTTP Requests
Content: [Documentation on Client usage]
```

**LLM Output**:
```markdown
The `Client` class in httpx makes HTTP requests through several convenience methods...

[client.py:142-158] The `get()` method is implemented as a wrapper around the internal `_request()` method...

Example usage:
```python
client = Client()
response = client.get("https://example.com")
```

[API.md:120-145] The documentation explains that...
```

---

## Configuration

### Constants

```python
# const.py additions

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================

# Retrieval parameters
TOP_K_PER_REQUEST = 20  # Fixed top-k per retrieval request
RRF_K = 60  # RRF constant for reciprocal rank calculation
RRF_TOP_N = 50  # Fetch top-N from each system before RRF
MAX_RETRIEVAL_REQUESTS = 5  # Limit router output to prevent explosion

# Weighted RRF (v1.1) - Favor semantic search over keyword matching
RRF_BM25_WEIGHT = 0.4  # Reduced weight for BM25 (prevents length bias)
RRF_VECTOR_WEIGHT = 1.0  # Full weight for FAISS vector search

# Folder boost (v1.2) - Soft filtering to preserve recall
RRF_FOLDER_BOOST = 1.3  # Chunks matching LLM-inferred folders get 1.3x boost

# Context expansion limits
MAX_RELATED_METHODS = 3  # Code: sibling methods to include
MAX_IMPORTS = 5  # Code: import statements to include
MAX_CHILDREN_SECTIONS = 10  # Markdown: child sections to list

# FAISS configuration
FAISS_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension
FAISS_INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# BM25 tokenization (v1.1 - Code-aware)
BM25_MIN_TOKEN_LENGTH = 3  # Ignore tokens shorter than this (except hex codes)
BM25_STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been"}
BM25_SPLIT_CAMELCASE = True  # Split camelCase identifiers (HTTPClient → http client)
BM25_SPLIT_SNAKE_CASE = True  # Split snake_case identifiers (get_user → get user)
BM25_KEEP_HEX_CODES = True  # Preserve hex codes like 0x884 as single tokens

# Router configuration
ROUTER_MAX_TREE_DEPTH = 4  # Max directory nesting in tree
ROUTER_MODEL = "gpt-4o-mini"  # Fast, cheap model for routing
ROUTER_TEMPERATURE = 0.1  # Low temperature for structured output

# Fast Path Router (v1.1) - Regex-based routing for simple queries
FAST_PATH_ROUTER_ENABLED = True  # Enable fast path for trivial queries
FAST_PATH_PATTERNS = {
    "hex_code": r'^0x[0-9A-Fa-f]+$',  # Match hex codes like 0x884
    "function_name": r'^[a-zA-Z_][a-zA-Z0-9_]*$',  # Match snake_case identifiers
    "camel_case_class": r'^[A-Z][a-zA-Z0-9]*$',  # Match PascalCase class names
    "file_name": r'^[a-zA-Z0-9_\-]+\.[a-z]{2,4}$',  # Match file.ext patterns
}

# Metadata Filtering Fallback (v1.1) - Graduated relaxation to prevent zero results
METADATA_FILTER_FALLBACK_ENABLED = True  # Enable graduated fallback strategy

# Synthesis configuration
SYNTHESIS_MODEL = "gpt-4o"  # High-quality model for final answer
SYNTHESIS_TEMPERATURE = 0.3  # Slightly creative for readability
SYNTHESIS_MAX_TOKENS = 4096  # Max output length
```

### Settings

```python
# settings.py additions

from const import (
    TOP_K_PER_REQUEST,
    RRF_K,
    FAISS_INDEX_DIR,
    EMBEDDING_MODEL,
    ROUTER_MODEL,
    SYNTHESIS_MODEL
)

# Create index directory
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Index file paths
CODE_FAISS_INDEX = FAISS_INDEX_DIR / "code_faiss.index"
MARKDOWN_FAISS_INDEX = FAISS_INDEX_DIR / "markdown_faiss.index"
TEXT_FAISS_INDEX = FAISS_INDEX_DIR / "text_faiss.index"

# Metadata cache path (pickled dict)
METADATA_CACHE_FILE = FAISS_INDEX_DIR / "metadata_cache.pkl"
```

---

## Future Enhancements

These features are **not implemented** in the initial version but are documented for future consideration:

### 1. Re-Retrieval with Evaluator

**Description**: Add evaluation step between retrieval and synthesis to assess if retrieved information is sufficient.

**Architecture**:
```
Retrieval → Evaluator (LLM) → [Re-Retrieve or Synthesize]
                              ↓
                        Generate new queries
```

**Evaluator Output Schema**:
```python
class EvaluatorDecision(BaseModel):
    decision: Literal["synthesize", "re_retrieve", "insufficient"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    missing_information: Optional[list[str]] = None  # If re_retrieve
    relevant_chunk_ids: Optional[list[str]] = None  # If synthesize
```

**Challenges**:
- **Latency**: +2-3s per evaluation loop
- **Loop risk**: Need strict exit conditions (max 2 re-retrievals)
- **Complexity**: More failure modes

**When to implement**: If single-pass retrieval success rate < 80%

---

### 2. Query Expansion

**Description**: Generate synonyms and related terms to improve recall.

**Example**:
```python
Original query: "BM25 ranking"
Expanded: "BM25 ranking OR Okapi BM25 OR probabilistic ranking"
```

**Trade-offs**:
- ✅ Better recall for terminology variations
- ❌ More noise in results
- ❌ Slower retrieval (larger result set)

**When to implement**: If queries with synonyms fail to retrieve relevant chunks

---

### 3. Cross-File Dependency Tracking

**Description**: Automatically fetch related classes from imported modules.

**Example**:
```python
# client.py imports Transport
from .transport import Transport

# Context expansion automatically fetches Transport class
```

**Challenges**:
- Context explosion (need strict limits)
- Circular dependencies
- Performance impact

**When to implement**: If users frequently ask about cross-file interactions

---

### 4. Adaptive Top-K

**Description**: Let router specify top_k per request based on query specificity.

**Example**:
```python
# Specific query
{"query": "README.md installation section", "top_k": 5}

# Broad query
{"query": "error handling patterns in codebase", "top_k": 30}
```

**Trade-offs**:
- ✅ More flexibility
- ❌ Less predictable behavior
- ❌ Harder to debug

**When to implement**: If fixed top_k causes issues (too many/few results)

---

### 5. Semantic Caching

**Description**: Cache retrieval results for similar queries.

**Architecture**:
```python
class SemanticCache:
    def __init__(self):
        self.cache = {}  # query_embedding → retrieval_results

    def get(self, query: str, similarity_threshold: float = 0.95):
        query_emb = embed(query)
        for cached_emb, results in self.cache.items():
            if cosine_similarity(query_emb, cached_emb) > similarity_threshold:
                return results
        return None
```

**When to implement**: If same/similar queries are frequent (e.g., chatbot)

---

## Implementation Notes

### Indexing Pipeline

**Script**: `scripts/build_retrieval_index.py`

**Steps**:
1. Load processed chunks (JSON files)
2. Generate embeddings for all chunks
3. Build FAISS indices (one per source_type)
4. Save FAISS indices to disk
5. Build metadata cache (in-memory dict)
6. Pickle metadata cache to disk

**Pseudocode**:
```python
def build_retrieval_index():
    # Load chunks
    code_chunks = load_json(CODE_CHUNKS_FILE)
    markdown_chunks = load_json(MARKDOWN_CHUNKS_FILE)
    text_chunks = load_json(TEXT_CHUNKS_FILE)

    # Generate embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)

    for chunk in code_chunks:
        embedding = model.encode(chunk["content"])
        faiss_code.add(embedding)
        metadata_index["code"].append(ChunkMetadata(...))

    # Repeat for markdown and text

    # Save FAISS indices
    faiss.write_index(faiss_code, str(CODE_FAISS_INDEX))
    faiss.write_index(faiss_markdown, str(MARKDOWN_FAISS_INDEX))
    faiss.write_index(faiss_text, str(TEXT_FAISS_INDEX))

    # Save metadata cache
    with open(METADATA_CACHE_FILE, "wb") as f:
        pickle.dump(metadata_index, f)
```

---

### Module Structure

```
src/
├── retrieval/
│   ├── __init__.py
│   ├── router.py              # Router component
│   ├── metadata_filter.py     # Metadata filtering
│   ├── hybrid_retriever.py    # BM25 + FAISS + RRF
│   ├── context_expander.py    # Context expansion
│   └── chunk_loader.py        # Load chunks from JSON
├── synthesis/
│   ├── __init__.py
│   └── synthesizer.py         # Synthesis prompt + LLM call
└── storage/
    ├── __init__.py
    ├── faiss_store.py         # FAISS index management
    └── metadata_cache.py      # In-memory metadata index

models/
└── retrieval.py               # Pydantic models for retrieval

scripts/
└── build_retrieval_index.py   # Index building script
```

---

### Testing Strategy

#### **Unit Tests**

```python
# tests/test_metadata_filter.py
def test_filter_by_source_type():
    """Test filtering by single source_type."""

def test_filter_by_multiple_source_types():
    """Test filtering by multiple source_types."""

def test_filter_by_folder():
    """Test folder-based filtering."""

def test_filter_by_file_pattern():
    """Test file pattern matching with wildcards."""

# tests/test_hybrid_retriever.py
def test_bm25_search():
    """Test BM25 search on small corpus."""

def test_faiss_search():
    """Test FAISS vector search."""

def test_rrf_ranking():
    """Test RRF fusion with k=60."""

# tests/test_context_expander.py
def test_code_context_expansion():
    """Test parent class and sibling method expansion."""

def test_markdown_context_expansion():
    """Test header hierarchy expansion."""
```

#### **Integration Tests**

```python
# tests/test_retrieval_pipeline.py
def test_end_to_end_retrieval():
    """Test full pipeline: router → retrieval → expansion."""
    query = "How does Client.get() work?"
    results = retrieval_pipeline.execute(query)
    assert len(results) > 0
    assert results[0].chunk.source_type == "code"
```

---

### Performance Benchmarks

**Target Metrics** (125 files, ~1000 chunks):

| Stage | Latency | Notes |
|-------|---------|-------|
| Router | 1-3s | LLM call with tree diagram |
| Metadata Filtering | <10ms | In-memory dict lookup |
| BM25 Search | 50-100ms | rank_bm25 on filtered corpus |
| FAISS Search | 100-200ms | Disk load + search |
| RRF Ranking | <10ms | CPU-bound, fast |
| Chunk Loading | 50-100ms | JSON file reads |
| Context Expansion | 100-200ms | Metadata lookups |
| Synthesis | 2-5s | LLM call with full context |
| **Total** | **5-8s** | Single-pass retrieval |

---

## Glossary

- **BM25**: Best Match 25, probabilistic ranking function for keyword search
- **FAISS**: Facebook AI Similarity Search, vector indexing library
- **RRF**: Reciprocal Rank Fusion, algorithm for combining search rankings
- **Metadata Index**: Lightweight in-memory cache of chunk metadata (no content)
- **Context Expansion**: Enriching retrieved chunks with parent/sibling/import context
- **Router**: LLM component that decomposes queries into structured retrieval requests
- **Top-K**: Number of top-ranked results to return (K=20)
- **Source Type**: High-level chunk category (code/markdown/text)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.2 | 2025-12-07 | Soft folder filtering (boost, not exclude), RRF_FOLDER_BOOST setting |
| 1.1 | 2025-01-05 | Weighted RRF, code-aware BM25 tokenization, fast path router |
| 1.0 | 2025-01-05 | Initial PRD (hybrid retrieval, metadata filtering, context expansion) |

---

## References

- **BM25 Algorithm**: Robertson & Zaragoza (2009) "The Probabilistic Relevance Framework: BM25 and Beyond"
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **RRF Paper**: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- **Chunking Architecture**: `documents/CHUNKING_ARCHITECTURE.md`
