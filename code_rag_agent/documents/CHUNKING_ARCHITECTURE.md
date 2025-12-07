# Code RAG Agent - Chunking Architecture

**Version:** 2.1
**Author:** Code RAG Agent Team
**Last Updated:** 2025-12-06

---

## Executive Summary

This document specifies the chunking architecture for the Code RAG Agent. The system implements a **unified chunk schema** with **type-specific processors** that transform heterogeneous source files into semantically meaningful, retrieval-optimized units.

### Design Philosophy

The chunking layer is the **foundation of retrieval quality**. Poor chunking decisions propagate downstream—breaking code semantics, severing document context, or creating chunks too large for precise retrieval. This architecture prioritizes:

1. **Semantic Integrity**: Never split mid-function or mid-section
2. **Context Preservation**: Chunks carry enough metadata to reconstruct meaning
3. **Retrieval Optimization**: Right-sized chunks for hybrid BM25 + vector search
4. **Type Safety**: Schema violations caught at creation time, not query time

---

## Architecture Overview

```
                              +------------------+
                              |  Source Files    |
                              |  (Repository)    |
                              +--------+---------+
                                       |
           +---------------------------+---------------------------+
           |                           |                           |
           v                           v                           v
+=====================+    +=====================+    +=====================+
|   CODE PROCESSOR    |    | MARKDOWN PROCESSOR  |    |   TEXT PROCESSOR    |
|                     |    |                     |    |                     |
| - AST parsing       |    | - Header splitting  |    | - Line-based split  |
| - Semantic units    |    | - Hierarchy inherit |    | - Size constraints  |
| - Import extraction |    | - Code block detect |    | - Extension metadata|
+=====================+    +=====================+    +=====================+
           |                           |                           |
           +---------------------------+---------------------------+
                                       |
                                       v
                    +======================================+
                    |        UNIFIED CHUNK SCHEMA          |
                    |                                      |
                    |  BaseChunk (Pydantic v2)             |
                    |  - Core fields (id, content, etc.)   |
                    |  - Type-specific optional fields     |
                    |  - Validation at creation time       |
                    +======================================+
                                       |
                                       v
                    +--------------------------------------+
                    |         OUTPUT (JSON Files)          |
                    |                                      |
                    | code_chunks.json     (~500 chunks)   |
                    | markdown_chunks.json (~200 chunks)   |
                    | text_chunks.json     (~50 chunks)    |
                    +--------------------------------------+
```

---

## Design Decisions

### Decision 1: AST-Based Code Chunking vs Line-Based

**Problem:** How to split Python files into retrieval units?

**Options Evaluated:**

| Approach | Semantic Integrity | Implementation | Retrieval Quality |
|----------|-------------------|----------------|-------------------|
| Line-based (every N lines) | ❌ Breaks functions | Simple | Poor |
| Regex-based (def/class) | ⚠️ Misses edge cases | Medium | Medium |
| **AST-based parsing** | ✅ Perfect boundaries | Complex | **Excellent** |

**Decision:** AST-based parsing using Python's `ast` module.

**Rationale:**
- Functions and classes are the **natural semantic units** of code
- AST guarantees syntactically valid boundaries (never splits mid-expression)
- Enables rich metadata extraction (docstrings, imports, parent class)
- Worth the implementation complexity for retrieval quality

**Trade-off Accepted:** AST parsing adds ~50ms per file vs ~5ms for line-based. Acceptable for offline indexing.

**Implementation:** [chunk_code.py](../scripts/data_pipeline/processing/chunk_code.py)

---

### Decision 2: Signature-Only Class Chunks

**Problem:** Class definitions can be 500+ tokens. Including full method bodies creates:
- Chunks too large for precise retrieval
- Duplicate content (methods appear in both class and method chunks)

**Options Evaluated:**

| Approach | Avg Tokens | Duplication | Retrieval Precision |
|----------|-----------|-------------|---------------------|
| Full class with all methods | 500-800 | High | Low (too broad) |
| Class excluded (methods only) | N/A | None | Low (no overview) |
| **Signature-only class** | ~150 | None | **High** |

**Decision:** Class chunks contain only signatures (definition + docstring + method signatures without bodies).

**Example:**

```python
# Original class (500+ tokens)
class HTTPClient:
    """Client for HTTP requests."""

    def get(self, url: str) -> Response:
        """Send GET request."""
        # 20 lines of implementation...

    def post(self, url: str, data: dict) -> Response:
        """Send POST request."""
        # 25 lines of implementation...

# Class chunk (~100 tokens) - Signature only
class HTTPClient:
    """Client for HTTP requests."""
    def get(self, url: str) -> Response: ...
    def post(self, url: str, data: dict) -> Response: ...

# Method chunks (separate, full implementation)
# chunk_1: get() with full body
# chunk_2: post() with full body
```

**Benefits:**
- ✅ Class chunk provides "API overview" for structural queries
- ✅ Method chunks provide implementation details
- ✅ No code duplication across chunks
- ✅ 70% reduction in class chunk size

---

### Decision 3: Unified Schema vs Separate Schemas

**Problem:** Three document types (code, markdown, text) have different metadata needs. How to model?

**Options Evaluated:**

| Approach | Query Simplicity | Type Safety | Schema Evolution |
|----------|-----------------|-------------|------------------|
| Separate collections | Complex (3 queries) | High | 3x migration effort |
| Union types | Medium | High | Complex validation |
| **Unified with discriminator** | Simple (1 query) | High | Single migration |

**Decision:** Single `BaseChunk` schema with `source_type` discriminator and optional type-specific fields.

**Schema Design:**

```python
class BaseChunk(BaseModel):
    # ═══ CORE (Required for all types) ═══
    id: str                    # MD5 hash
    source_type: Literal["code", "markdown", "text"]
    chunk_type: str            # Subtype validation via source_type
    content: str
    token_count: int
    file_path: str
    start_line: int
    end_line: int
    name: str

    # ═══ CODE-SPECIFIC (Optional) ═══
    docstring: str | None = None
    imports: list[str] = []
    parent_context: str = ""

    # ═══ MARKDOWN-SPECIFIC (Optional) ═══
    headers: dict[str, str] = {}
    heading_level: int | None = None

    # ═══ TEXT-SPECIFIC (Optional) ═══
    file_extension: str | None = None
```

**Query Comparison:**

```python
# Unified (chosen) - Single query, filter by type
chunks = db.query(source_type="code", chunk_type="function")

# Separate (rejected) - Multiple collections, complex unions
code = code_db.query(chunk_type="function")
md = markdown_db.query(chunk_type="section")
all_chunks = merge(code, md)  # Complex aggregation
```

**Rationale:**
- ✅ Simple querying: One collection, one filter
- ✅ Type safety: Pydantic validates at creation
- ✅ Future-proof: New source types = add fields, not migrations
- ✅ Storage overhead negligible (~5% for nullable fields)

**Implementation:** [chunk.py](../models/chunk.py)

---

### Decision 4: Two-Pass Markdown Processing

**Problem:** Markdown has two competing constraints:
1. **Semantic structure**: Headers create hierarchy that provides context
2. **Size limits**: Some sections exceed 1000 tokens (too large for retrieval)

**Options Evaluated:**

| Approach | Hierarchy Preserved | Size Control | Complexity |
|----------|--------------------|--------------| -----------|
| Header split only | ✅ Yes | ❌ No | Low |
| Size split only | ❌ No | ✅ Yes | Low |
| **Two-pass (header → size)** | ✅ Yes | ✅ Yes | Medium |

**Decision:** Two-pass processing with metadata inheritance.

**Algorithm:**

```
PASS 1: Header-Based Splitting
├── Split on H1, H2, H3, H4 headers
├── Extract header hierarchy as metadata
├── Filter out Table of Contents sections
└── Track original line numbers

PASS 2: Size-Constrained Splitting (if needed)
├── For each section > 1000 tokens:
│   ├── Apply RecursiveCharacterTextSplitter
│   ├── Inherit ALL metadata from parent
│   └── Update chunk_type to "markdown_section_chunk"
└── Preserve code block integrity (never split mid-fence)
```

**Metadata Inheritance Example:**

```markdown
# Getting Started           ← H1
## Installation             ← H2
### MacOS                   ← H3 (section > 1000 tokens, will be split)
[2000 tokens of content...]
```

**Output chunks (after splitting):**

```python
# Chunk 1 (first half of MacOS section)
{
    "headers": {"h1": "Getting Started", "h2": "Installation", "h3": "MacOS"},
    "chunk_type": "markdown_section_chunk",  # Indicates it was split
    "content": "[first 800 tokens...]"
}

# Chunk 2 (second half, inherits FULL header context)
{
    "headers": {"h1": "Getting Started", "h2": "Installation", "h3": "MacOS"},
    "chunk_type": "markdown_section_chunk",
    "content": "[remaining 1200 tokens...]"
}
```

**Rationale:**
- ✅ Header context prevents "orphaned" chunks
- ✅ LangChain's splitters are battle-tested
- ✅ Code blocks never split mid-fence (RecursiveCharacterTextSplitter respects markers)

**Implementation:** [process_markdown.py](../scripts/data_pipeline/processing/process_markdown.py)

---

### Decision 5: Pre-Computed Token Counts

**Problem:** Context window management requires knowing chunk sizes. When to compute?

**Options Evaluated:**

| Approach | Indexing Cost | Query Cost | Consistency |
|----------|--------------|------------|-------------|
| On-demand (query time) | None | High (tokenize each query) | Varies by tokenizer |
| **Pre-computed (index time)** | One-time | Zero | Consistent |

**Decision:** Pre-compute `token_count` during indexing and store in chunk metadata.

**Use Cases Enabled:**

```python
# 1. Context window budgeting
def select_chunks_for_context(chunks, max_tokens=4000):
    selected = []
    budget = max_tokens
    for chunk in chunks:
        if chunk.token_count <= budget:
            selected.append(chunk)
            budget -= chunk.token_count
    return selected

# 2. Retrieval filtering (prefer smaller chunks)
chunks.filter(token_count__lte=500)

# 3. Analytics
avg_tokens = sum(c.token_count for c in chunks) / len(chunks)
```

**Rationale:**
- ✅ Zero query-time overhead
- ✅ Consistent estimates (same algorithm for all chunks)
- ✅ Enables budget-aware retrieval
- ✅ Storage cost: 4 bytes per chunk (negligible)

---

### Decision 6: Chunk ID Generation Strategy

**Problem:** Chunks need unique, stable identifiers. What to use?

**Options Evaluated:**

| Approach | Uniqueness | Stability | Debuggability |
|----------|-----------|-----------|---------------|
| UUID | ✅ Unique | ❌ Changes on re-index | ❌ Opaque |
| Sequential (1, 2, 3...) | ⚠️ Collision risk | ❌ Order-dependent | ⚠️ Limited |
| **Content hash (MD5)** | ✅ Unique | ✅ Deterministic | ✅ Traceable |

**Decision:** MD5 hash of `file_path + start_line + content_preview`.

**Implementation:**

```python
def create_chunk_id(file_path: str, start_line: int, content: str) -> str:
    """Generate deterministic chunk ID."""
    preview = content[:100]  # First 100 chars for uniqueness
    raw = f"{file_path}:{start_line}:{preview}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]
```

**Benefits:**
- ✅ Same content → same ID (enables incremental updates)
- ✅ ID encodes location information (debuggable)
- ✅ Collision probability: ~0 for <1M chunks (birthday paradox)

---

## Data Flow

### End-to-End Processing Pipeline

```
1. REPOSITORY CLONE
   git clone httpx → data/httpx/
        |
        v
2. FILE DISCOVERY
   Glob patterns:
   - *.py → Code processor
   - *.md → Markdown processor
   - *.toml, *.yaml, *.json, *.txt → Text processor
        |
        v
3. PARALLEL PROCESSING
   +------------------+------------------+------------------+
   |  Code Processor  |   MD Processor   |  Text Processor  |
   |                  |                  |                  |
   |  Files: 85       |   Files: 32      |   Files: 8       |
   |  Chunks: 520     |   Chunks: 180    |   Chunks: 45     |
   |  Time: ~4s       |   Time: ~1s      |   Time: ~0.2s    |
   +------------------+------------------+------------------+
        |                     |                    |
        v                     v                    v
4. VALIDATION (Pydantic v2)
   - Schema compliance check
   - Type-specific field validation
   - chunk_type ↔ source_type consistency
        |
        v
5. SERIALIZATION
   +--------------------------------------------------+
   |  data/processed/                                  |
   |  ├── code_chunks.json      (520 chunks, ~5MB)    |
   |  ├── markdown_chunks.json  (180 chunks, ~2MB)    |
   |  ├── text_chunks.json      (45 chunks, ~0.5MB)   |
   |  └── all_chunks.json       (745 chunks, ~7.5MB)  |
   +--------------------------------------------------+
```

### Code Processing Detail

```
Python File (e.g., client.py)
        |
        v
+---------------------------+
| 1. ast.parse(source)      |
|    Parse into AST tree    |
+---------------------------+
        |
        v
+---------------------------+
| 2. Extract Imports        |
|    Filter: top-level only |
|    Skip: __future__       |
+---------------------------+
        |
        v
+---------------------------+
| 3. Process Classes        |
|    For each ClassDef:     |
|    ├── Build signature    |
|    │   (def + doc + sigs) |
|    ├── Create class chunk |
|    └── Create method chunks|
+---------------------------+
        |
        v
+---------------------------+
| 4. Process Functions      |
|    For each FunctionDef:  |
|    ├── Extract full code  |
|    ├── Extract docstring  |
|    └── Create chunk       |
+---------------------------+
        |
        v
+---------------------------+
| 5. Enrich Metadata        |
|    ├── Generate ID (MD5)  |
|    ├── Count tokens       |
|    ├── Set source_type    |
|    └── Extract filename   |
+---------------------------+
        |
        v
   list[BaseChunk]
```

---

## Component Specifications

### File Structure

| Component | File | Responsibility |
|-----------|------|----------------|
| **BaseChunk Model** | [models/chunk.py](../models/chunk.py) | Pydantic schema with validation |
| **Code Processor** | [chunk_code.py](../scripts/data_pipeline/processing/chunk_code.py) | AST-based Python chunking |
| **Markdown Processor** | [process_markdown.py](../scripts/data_pipeline/processing/process_markdown.py) | Header-aware splitting |
| **Text Processor** | [process_text_files.py](../scripts/data_pipeline/processing/process_text_files.py) | Line-based splitting |
| **Orchestrator** | [process_all_files.py](../scripts/data_pipeline/processing/process_all_files.py) | Runs all processors |
| **Constants** | [const.py](../const.py) | SOURCE_TYPES, CHUNK_TYPE_MAP |

### Configuration Constants

```python
# const.py - Chunking Configuration

# Source type definitions
SOURCE_TYPES: list[str] = ["code", "markdown", "text"]

# Valid chunk types per source
CHUNK_TYPE_MAP = {
    "code": ["function", "class"],
    "markdown": ["markdown_section", "markdown_section_chunk"],
    "text": ["text_file", "text_file_chunk"]
}

# File extensions
CODE_EXTENSIONS: set[str] = {'.py'}
TEXT_EXTENSIONS: set[str] = {'.toml', '.yml', '.yaml', '.txt', '.ini', '.json'}

# Size constraints
DEFAULT_MAX_CHUNK_SIZE = 2000      # characters
DEFAULT_CHUNK_OVERLAP = 200        # characters
MAX_CHUNK_SIZE_MARKDOWN = 1000    # tokens (triggers second-pass split)
```

---

## Validation Strategy

### Pydantic v2 Enforcement

Schema violations are caught at **creation time**, not query time:

```python
# ❌ FAILS - Invalid chunk_type for source_type
BaseChunk(
    source_type="code",
    chunk_type="markdown_section"  # ValidationError!
)

# ❌ FAILS - Missing required field
BaseChunk(
    source_type="code",
    chunk_type="function"
    # Missing: content, file_path, etc.
)

# ✅ PASSES - Valid chunk
BaseChunk(
    id="a3f5e9b2c1d4",
    source_type="code",
    chunk_type="function",
    content="def get(self): ...",
    token_count=15,
    file_path="client.py",
    start_line=42,
    end_line=44,
    name="get"
)
```

**Benefits:**
- Fail fast: Errors caught during indexing, not retrieval
- Type safety: IDE autocomplete, static analysis
- Self-documenting: Models serve as schema reference

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Code processing** | ~50ms/file | AST parsing is bottleneck |
| **Markdown processing** | ~30ms/file | Regex in header split |
| **Text processing** | ~10ms/file | Simple line iteration |
| **Total (125 files)** | ~5-6 seconds | Parallelizable |
| **Output size** | ~7.5MB JSON | Compressed: ~1.5MB |
| **Avg chunk tokens** | 350-500 | Optimal for retrieval |

---

## Testing

All chunking functionality validated with **16 passing tests**:

```bash
pytest tests/test_processing_pipeline.py -v

# Test Coverage:
# - Code: Signature extraction, import filtering, class compression
# - Markdown: Header hierarchy, code block integrity, TOC filtering
# - Schema: Required fields, token counts, line numbers
# - Utils: Token estimation, edge cases
```

---

## References

- **Chunking Architecture**: This document
- **Retrieval Architecture**: [RETRIEVAL_ARCHITECTURE.md](RETRIEVAL_ARCHITECTURE.md)
- **Building Blocks**: [BUILDING_BLOCKS_ARCHITECTURE.md](BUILDING_BLOCKS_ARCHITECTURE.md)
- **LangChain Text Splitters**: https://python.langchain.com/docs/modules/data_connection/document_transformers/
