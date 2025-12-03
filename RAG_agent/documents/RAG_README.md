# RAG (Retrieval-Augmented Generation) System Documentation

## Overview

This RAG system implements hybrid search combining vector similarity, keyword matching (BM25), and optional web search for comprehensive document retrieval from LangChain, LangGraph, and LangSmith documentation.

## Architecture

```
User Query
    ↓
[Intent Classification] ← Classifies query intent and extracts metadata
    ↓
[Hybrid Retrieval] ← Combines 2 search methods in parallel
    ├── Vector Search (ChromaDB with HNSW)
    ├── BM25 Keyword Search
    └── Web Search (Tavily API, online mode only - optional)
    ↓
[Deduplication] ← Removes duplicate text content
    ↓
[Hybrid Scoring] ← Weighted combination of scores
    ↓
[Metadata Filtering] ← Filter by framework/language
    ↓
[Top-5 Results] → Agent Response Generation
```

## Content Granularity Levels

The system uses **3-level granular retrieval** for precise content matching:

### Level 1: Document-Level
- **What**: Entire document with all sections
- **When**: BM25 matches or vector search finds document-level embedding
- **Use Case**: Broad overviews, conceptual understanding
- **Example**: "What is LangChain?"

### Level 2: Section-Level
- **What**: Single section with title and content blocks
- **When**: Vector search finds section-level embedding
- **Use Case**: Specific topics within a document
- **Example**: "How does memory work in LangGraph?"

### Level 3: Content-Block-Level (Most Common)
- **What**: Specific paragraph, code snippet, or list item
- **When**: Vector search finds content-block-level embedding (80%+ of queries)
- **Use Case**: Precise answers, code examples, specific instructions
- **Example**: "Show me code for creating a ReAct agent"

## Content Expansion Strategy

When a **content-block** matches (Level 3), the system automatically expands to include the **full section** (Level 2) to provide complete context.

### What Gets Included ✅

**Always Included:**
- The matched content block itself (most relevant)
- All other content blocks in the same section
- Section title and metadata
- Document source information

### What Gets Excluded ❌

**Never Included (to avoid noise):**
- Parent sections (too broad, loses focus)
- Sibling sections (often unrelated topics)
- Child subsections (already included if relevant)
- Cross-document links (removed in recent refactor for performance)

### Example

Given this document structure:
```
Document: "Memory in LangChain"
├── Section 1: "Overview" (level 1)
│   ├── Block 1.1: "Memory allows agents to remember..."
│   ├── Block 1.2: "There are two types of memory:"
│   └── Block 1.3: "Short-term memory stores recent messages."
├── Section 2: "Short-term memory" (level 2)  ← MATCHED SECTION
│   ├── Block 2.1: "Short-term memory is implemented via..."
│   ├── Block 2.2: "Example: memory = ChatMessageHistory()"  ← MATCHED BLOCK
│   └── Block 2.3: "To use short-term memory, add it to your agent."
└── Section 3: "Long-term memory" (level 2)
    └── ...
```

**Query**: "How do I implement short-term memory?"

**Match**: Block 2.2 (code example)

**Returned Content**:
```markdown
## Section: Short-term memory

[Block 2.1] Short-term memory is implemented via...
[Block 2.2] Example: memory = ChatMessageHistory()
[Block 2.3] To use short-term memory, add it to your agent.
```

**NOT Included**:
- ❌ Section 1 (parent section - too broad)
- ❌ Section 3 (sibling section - different topic)
- ❌ Other documents about memory

## Rationale

**Why expand content blocks to full sections?**
- Content blocks are often small (1-2 sentences)
- Without surrounding context, they may be incomplete or confusing
- Full section provides complete explanation with examples
- Prevents "orphaned" snippets that lack context

**Why NOT expand further (parents/siblings)?**
- Parent/sibling sections are often too broad or unrelated
- More content = more noise for the LLM to process
- Harder for LLM to find the specific answer
- Better to retrieve multiple diverse documents than one giant document

**When expansion might be insufficient:**
- Very long sections (>2000 chars) get truncated in `agent_response.py`
- Deeply nested sections might lack broader context
- Solution: Trust vector search to find the most relevant granularity level

## Hybrid Scoring Formula

Results from all search methods are combined using weighted scores:

```python
final_score = (vector_score * 0.7) + (bm25_score * 0.3) + web_bonus(0.2)
```

**Weights Rationale:**
- **Vector (70%)**: Semantic similarity is most reliable for conceptual matches
- **BM25 (30%)**: Keyword matching ensures important terms are covered
- **Web Bonus (+20%)**: Fresh content from web gets small boost (online mode only)

**Score Normalization:**
- Vector scores: 0-1 range (cosine similarity)
- BM25 scores: Normalized to 0-1 (max BM25 ≈ 50)
- Web scores: 0-1 range from Tavily API
- Final relevance score: 0-100 scale for display

## Deduplication Strategy

The system applies **two levels of deduplication**:

### 1. Text-Based Deduplication (Critical)
**Location**: `hybrid_retrieval_vector.py:_deduplicate_by_text()`

**Problem Solved**: Identical content appears in multiple file paths
- `langchain/python/learn.json`
- `langchain/oss/python/learn.json`
- `langchain/javascript/learn.json`

All contain identical "LangGraph" sections.

**Solution**:
```python
# Normalize text: lowercase, collapse whitespace
normalized = ' '.join(text.lower().split())

# Skip if exact match already seen
if normalized in seen_texts:
    skip_duplicate()
```

**Result**: 40% reduction in duplicates (30 → 22 results in tests)

### 2. Section-Level Deduplication (Vector Search)
**Location**: `vector_search.py:_deduplicate_by_section()`

**Problem Solved**: Multiple content blocks from same section match

**Solution**: Keep only highest-scoring block per `(document_path, section_title)` tuple

**Result**: Prevents repetitive section content in results

### 3. Domain-Based Deduplication (Web Search)
**Location**: `deduplicator.py:deduplicate_web_results()`

**Problem Solved**: Web results duplicate offline documentation

**Solution**: Remove web results from known offline domains:
- `python.langchain.com`
- `docs.langchain.com`
- `js.langchain.com`
- etc.

**Result**: Web search supplements (not duplicates) offline docs

## Search Methods

### 1. Vector Search (ChromaDB + HNSW)

**Technology**: ChromaDB with HNSW (Hierarchical Navigable Small World) indexing

**Embedding Model**:
- Default: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- Alternative: `models/embedding-001` (Google Gemini, requires API key)

**Index Size**: ~19,000 embeddings across 1,048 documents

**Features**:
- Semantic similarity matching
- 3-level granularity (document, section, content-block)
- Metadata filtering (framework, language, topic)
- Granularity-based reranking (prefers finer-grained matches)

**Performance**: ~100-300ms per query

### 2. BM25 Keyword Search

**Technology**: `rank_bm25` library (Okapi BM25 algorithm)

**Index**: Full-text index of all documents (title + sections)

**Features**:
- Keyword/lexical matching
- TF-IDF based scoring
- Fast lookups (~10-50ms)

**Use Case**: Ensures queries with specific technical terms get matched

### 3. Web Search (Online Mode Only)

**Technology**: Tavily API

**Features**:
- Intelligent query generation (from user intent)
- Up-to-date web content
- Deduplicated against offline docs
- 3 results per query max

**Performance**: ~500-2000ms (external API call)

**Configuration**: Set `AGENT_MODE=online` and `TAVILY_API_KEY` in `.env`

## Metadata Filtering

Results are filtered by user intent metadata:

**Framework Filter**:
- `langchain`: LangChain framework docs
- `langgraph`: LangGraph framework docs
- `langsmith`: LangSmith framework docs
- `general`: No filter (all frameworks)
- `web`: Always kept (assumed relevant)

**Language Filter**:
- `python`: Python-specific docs
- `javascript`: JavaScript-specific docs
- `unknown`: Language-agnostic docs (always kept)

**Filter Logic**:
```python
if intent.framework == "langgraph":
    keep only (framework == "langgraph" OR framework == "web")

if intent.language == "python":
    keep only (language == "python" OR language in ["unknown", ""] OR framework == "web")
```

## Performance Optimizations

### 1. Async/Await for Parallel Search
All 3 search methods run concurrently using `asyncio.gather()`:
```python
vector_results, bm25_results, web_results = await asyncio.gather(
    _run_vector_search_async(query),
    _run_bm25_search_async(query),
    _run_web_search_async(query),
)
```

**Benefit**: Total latency = max(vector, bm25, web) instead of sum

### 2. Eager Component Loading
RAG components (index, BM25, VectorDB) are loaded once at startup:
- Reduces per-query latency
- Amortizes loading cost across all queries
- Components persist in memory

### 3. Removed Link Fetching
**Before**: Fetched content from external links (500ms-5s overhead)
**After**: Removed for 40% latency reduction (2.0s → 1.2s avg)

### 4. Simplified Scoring Logic
**Before**: 6 functions, 260 lines, mixed data structures
**After**: 1 `HybridScorer` class, 150 lines, consistent data model

**Benefit**: Easier to maintain, debug, and optimize

## Configuration

### Environment Variables (`.env`)

```bash
# Required
GOOGLE_API_KEY=your_gemini_key_here

# Embedding Model (choose one)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Local (default)
# EMBEDDING_MODEL=models/embedding-001  # Google Gemini (requires API key)

# Agent Mode
AGENT_MODE=offline  # Use only local docs (default)
# AGENT_MODE=online  # Enable web search (requires TAVILY_API_KEY)

# Web Search (online mode only)
TAVILY_API_KEY=your_tavily_key_here
WEB_SEARCH_MAX_RESULTS=5
```

### Tuning Scoring Weights

Edit `src/rag/hybrid_scorer.py`:
```python
# Scoring weights (tuned empirically)
VECTOR_WEIGHT = 0.7   # Semantic similarity
BM25_WEIGHT = 0.3     # Keyword coverage
WEB_BONUS = 0.2       # Web content boost
```

**Guidelines**:
- Increase `VECTOR_WEIGHT` for more semantic matching
- Increase `BM25_WEIGHT` for more keyword matching
- Adjust `WEB_BONUS` to prioritize/deprioritize web results

## File Structure

```
src/rag/
├── __init__.py
├── README.md (this file)
├── vector_search.py          # ChromaDB vector search
├── bm25_search.py            # BM25 keyword search
├── hybrid_scorer.py          # Hybrid scoring logic (NEW)
├── document_index.py         # In-memory document index
├── deduplicator.py           # Web result deduplication
├── web_search_api.py         # Tavily API wrapper
└── web_search_query_generator.py  # Web query generation

src/nodes/
└── hybrid_retrieval_vector.py  # Main retrieval orchestration node

data/
├── rag_components/
│   ├── document_index.pkl    # Pre-built document index
│   └── bm25_index.pkl        # Pre-built BM25 index
└── vector_db/                # ChromaDB persistent storage
```

## Building RAG Components

**First time setup**:
```bash
# Build document index, BM25 index, and vector embeddings
python scripts/data_pipeline/indexing/build_rag_components.py
```

**This creates**:
- `data/rag_components/document_index.pkl` (~10MB)
- `data/rag_components/bm25_index.pkl` (~15MB)
- `data/vector_db/` directory (~500MB)

**Rebuild when**:
- Documentation files change
- Switching embedding models
- Updating search parameters

## Testing

**Quick test** (standalone):
```bash
python test_rag_eval.py
```

**Full integration test**:
```bash
pytest tests/test_end_to_end_retrieval.py -v
```

**Test specific query**:
```python
from src.nodes.hybrid_retrieval_vector import initialize_rag_components, hybrid_retrieval_node
from models.intent_classification import IntentClassification

# Initialize components
initialize_rag_components()

# Build state
state = {
    "user_input": "How do I create a LangGraph agent?",
    "cleaned_request": "How do I create a LangGraph agent?",
    "intent_result": IntentClassification(
        intent_type="implementation_guide",
        framework="langgraph",
        language=None,
        keywords=["create", "agent"],
        topics=["agent creation"],
        requires_rag=True
    )
}

# Run retrieval
result = hybrid_retrieval_node(state)
docs = result["retrieved_documents"]

print(f"Retrieved {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"{i}. {doc.title} (score: {doc.relevance_score:.1f})")
```

## Metrics & Monitoring

**Key Metrics** (logged automatically):
- Retrieval latency (per query)
- Number of results per search method
- Deduplication counts
- Score distributions
- Granularity breakdown
- Source breakdown (offline vs web)

**Example log output**:
```
INFO - Text deduplication: 30 → 22 results (8 duplicates removed)
INFO - Hybrid scoring: 35 docs → 22 after filtering → returning top 5
INFO - Retrieved 5 results (avg score: 67.3)
INFO -   Sources: offline=5, web=0
INFO -   Granularity: {'content_block': 4, 'section': 1}
```

## Troubleshooting

### No results returned

**Check**:
1. Are RAG components built? (`data/rag_components/` exists?)
2. Is VectorDB populated? (`data/vector_db/` exists?)
3. Are filters too restrictive? (try `framework=None, language=None`)

**Solution**:
```bash
python scripts/data_pipeline/indexing/build_rag_components.py
```

### Duplicate results

**Check**: Is `_deduplicate_by_text()` being called?

**Verify** in logs:
```
INFO - Text deduplication: X → Y results (Z duplicates removed)
```

### Poor relevance

**Possible causes**:
1. Query too vague → Improve intent classification
2. Content not in docs → Enable web search (online mode)
3. Wrong framework filter → Check intent detection

**Debug**:
```python
# Print raw scores before ranking
logger.debug(f"Vector score: {match.score}")
logger.debug(f"BM25 score: {bm25_score}")
```

### Slow performance

**Check latency breakdown**:
- Vector search: Should be <300ms
- BM25 search: Should be <50ms
- Web search: Can be 500-2000ms

**If slow**:
1. Disable web search: `AGENT_MODE=offline`
2. Reduce `top_k` parameter
3. Use local embeddings (not Gemini API)

## Future Improvements

**Potential enhancements** (not currently implemented):

1. **Parent Context** (optional):
   - Include parent section for deeply nested content blocks (level >= 3)
   - Helps understand hierarchical context
   - Trade-off: Adds content length

2. **Fuzzy Deduplication**:
   - Use `rapidfuzz` for near-duplicate detection (95% similarity)
   - Catches minor text variations
   - Trade-off: May incorrectly deduplicate similar but distinct content

3. **Configurable Weights**:
   - Move scoring weights to `settings.py`
   - Allow runtime tuning via environment variables
   - A/B test different weight combinations

4. **Reranking Model**:
   - Add cross-encoder reranking (e.g., `ms-marco-MiniLM`)
   - Rerank top-25 to final top-5
   - Trade-off: Adds 100-200ms latency

5. **Query Rewriting**:
   - LLM-based query expansion/reformulation
   - Generate multiple query variations
   - Trade-off: LLM call overhead (~500ms)

## References

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [RAG Best Practices (LangChain)](https://python.langchain.com/docs/how_to/qa_chat_history_how_to)
- [Hybrid Search Strategies](https://www.pinecone.io/learn/hybrid-search-intro/)

---

**Last Updated**: 2025-12-02
**Version**: 2.0 (Post-refactor)
**Maintainer**: RAG Team
