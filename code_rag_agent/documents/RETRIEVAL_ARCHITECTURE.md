# Retrieval Architecture PRD

**Status**: Implementation Ready

---

## Overview

This document specifies the retrieval architecture for the Code RAG Agent. The system implements a **hybrid retrieval strategy** combining BM25 (lexical) and vector embeddings (semantic) with metadata filtering and context expansion.

For a beginner-friendly explanation of the implementation, see [FAISS_IMPLEMENTATION.md](FAISS_IMPLEMENTATION.md).

### Design Principles

1. **Hybrid Search**: Combine BM25 and FAISS vector search with RRF ranking
2. **Metadata-First Filtering**: Filter by source type, folders, and files before search
3. **Context Expansion**: Enrich retrieved chunks with parent/sibling context
4. **Router-Guided Retrieval**: LLM router decomposes queries into structured requests
5. **No Re-Retrieval**: Single-pass retrieval (evaluator reserved for future)

---

## High-Level Pipeline

```
User Query → Fast Path Router → [Hit: Direct search / Miss: LLM Router]
                                        ↓
                              RetrievalRequest(s)
                                        ↓
              ┌─────────────────────────────────────────┐
              │  For each request (parallel):          │
              │  1. Metadata Filter (source_type, file)│
              │  2. BM25 + FAISS search (parallel)     │
              │  3. RRF ranking + folder boost         │
              │  4. Top-K selection                    │
              └─────────────────────────────────────────┘
                                        ↓
                              Context Expansion
                                        ↓
                              Synthesis (LLM)
```

---

## Component Specifications

### 1. Router

#### Fast Path Router

Bypasses LLM for trivial queries using regex patterns:

```python
PATTERNS = {
    "hex_code": r'^0x[0-9A-Fa-f]+$',      # "0x884"
    "function_name": r'^[a-zA-Z_]\w*$',   # "get_user"
    "camel_case_class": r'^[A-Z][a-zA-Z]+$', # "HTTPClient"
    "file_name": r'^[\w\-]+\.[a-z]{2,4}$',  # "config.toml"
}
```

#### LLM Router

**Input**: User query + codebase tree
**Output**: `RouterOutput` with 1-5 `RetrievalRequest` objects

```python
class RetrievalRequest(BaseModel):
    query: str                    # Specific search query
    source_types: list[str]       # ["code", "markdown", "text"]
    folders: list[str]            # Soft boost (not hard filter)
    file_patterns: list[str]      # Hard filter (supports wildcards)
    reasoning: str                # Context for synthesis
```

---

### 2. Metadata Filtering

**Hard filters** (exclude non-matches):
- `source_types`: Which indices to search
- `file_patterns`: Filename wildcards (e.g., `*.py`, `client.py`)

**Soft filter** (boost, don't exclude):
- `folders`: Chunks matching get 1.3x RRF score boost

```python
def apply(self) -> list[str]:
    """Return chunk IDs matching source_type and file_patterns."""
    # Folders NOT filtered here - handled via score boost in RRF
```

---

### 3. Hybrid Search

#### Search Flow

```python
def search(self, request: RetrievalRequest, top_k: int = 20):
    # 1. Metadata filter → candidate_ids
    # 2. Parallel: BM25 + FAISS search (ThreadPoolExecutor)
    # 3. Weighted RRF: BM25=0.4, Vector=1.0
    # 4. Folder boost (1.3x for matches)
    # 5. Return top-k
```

#### RRF Formula

```
RRF_score(chunk) = 0.4 * (1/(k + rank_bm25)) + 1.0 * (1/(k + rank_faiss))
```

Where `k=60` (RRF constant).

#### Fallback Chain

If strict filtering yields 0 results:
1. Drop `file_patterns`
2. Drop `folders`
3. Emergency: search all `source_types`

---

### 4. Context Expansion

| Chunk Type | Context Added |
|------------|---------------|
| Code (function) | Parent class, sibling methods (max 3), imports (max 5) |
| Markdown | Parent header, child section list (max 10) |
| Text | None (standalone) |

---

## Configuration

### Key Settings (settings.py)

```python
# Retrieval
TOP_K_PER_REQUEST = 20
RRF_K = 60
RRF_TOP_N = 50
RRF_BM25_WEIGHT = 0.4
RRF_VECTOR_WEIGHT = 1.0
RRF_FOLDER_BOOST = 1.3

# Context Expansion
MAX_RELATED_METHODS = 3
MAX_IMPORTS = 5
MAX_CHILDREN_SECTIONS = 10

# FAISS
FAISS_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
FAISS_HNSW_M = 32
FAISS_HNSW_EF_CONSTRUCTION = 64
FAISS_HNSW_EF_SEARCH = 32
```

---

## Module Structure

```
src/retrieval/
├── chunk_loader.py      # Load chunks from JSON (singleton)
├── metadata_filter.py   # Pre-search filtering
├── faiss_store.py       # FAISS index management
├── hybrid_retriever.py  # BM25 + FAISS + RRF orchestration
├── context_expander.py  # Parent/sibling enrichment
└── fast_path_router.py  # Regex bypass for simple queries
```

---

## Future Enhancements

| Feature | Description | When to Implement |
|---------|-------------|-------------------|
| Re-Retrieval | Evaluator loop for insufficient results | If single-pass success < 80% |
| Query Expansion | Synonym generation | If synonym queries fail |
| Cross-File Dependencies | Auto-fetch imported classes | If users ask about cross-file interactions |
| Semantic Caching | Cache similar query results | If same queries are frequent |

---

## References

- [FAISS_IMPLEMENTATION.md](FAISS_IMPLEMENTATION.md) - Beginner-friendly implementation guide
- [CHUNKING_ARCHITECTURE.md](CHUNKING_ARCHITECTURE.md) - Chunk schema and processing
- BM25: Robertson & Zaragoza (2009)
- RRF: Cormack et al. (2009)
- FAISS: https://github.com/facebookresearch/faiss
