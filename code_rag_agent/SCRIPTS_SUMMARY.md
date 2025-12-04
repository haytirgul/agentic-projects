# Scripts Folder - Summary

## ✅ Complete Scripts Structure Created

Similar to `RAG_agent`, the `code_rag_agent` now has a complete `scripts/` folder with data pipeline and utilities.

### Directory Structure

```
scripts/
├── README.md                          # Complete documentation
├── __init__.py
│
├── data_pipeline/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── clone_httpx_repo.py       # Clone/update httpx repository
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   └── chunk_code.py             # AST-based code chunking
│   │
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── build_vector_index.py     # ChromaDB vector index
│   │   └── build_bm25_index.py       # BM25 keyword index
│   │
│   └── build_all_indices.py          # Master pipeline script
│
└── utilities/
    ├── __init__.py
    └── visualize_graph.py             # LangGraph visualization
```

---

## Scripts Overview

### 🔧 Data Pipeline Scripts

#### 1. **clone_httpx_repo.py** - Repository Ingestion
```bash
python scripts/data_pipeline/ingestion/clone_httpx_repo.py
```
- Clones `github.com/encode/httpx` to `data/httpx/`
- Updates if repository exists
- Shows commit info and file counts

#### 2. **chunk_code.py** - Code Processing
```bash
python scripts/data_pipeline/processing/chunk_code.py
```
- Parses Python files using AST
- Extracts functions, classes, methods
- Creates semantic chunks with metadata
- Output: `data/processed/code_chunks.json`

**Features**:
- AST-based chunking (preserves code structure)
- Extracts docstrings automatically
- Captures function/class context
- Includes imports for each chunk
- Respects `REPO_EXCLUDE_PATTERNS`

#### 3. **build_vector_index.py** - Semantic Search Index
```bash
python scripts/data_pipeline/indexing/build_vector_index.py
```
- Loads code chunks
- Generates embeddings via `sentence-transformers`
- Stores in ChromaDB for semantic search
- Output: `data/vector_db/` (ChromaDB persistent storage)

**Configuration**:
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch size: 32 (configurable)
- Collection: `code_chunks`

#### 4. **build_bm25_index.py** - Keyword Search Index
```bash
python scripts/data_pipeline/indexing/build_bm25_index.py
```
- Tokenizes code for keyword matching
- Builds BM25 index using `rank_bm25`
- Saves as pickle for fast loading
- Output: `data/rag_components/bm25_index.pkl`

**Features**:
- Code-aware tokenization
- Preserves Python identifiers (underscores)
- Fast retrieval for keyword queries

#### 5. **build_all_indices.py** - Master Pipeline ⭐
```bash
python scripts/data_pipeline/build_all_indices.py
```
- Runs complete pipeline in sequence
- Error handling and progress tracking
- Summary report with timing
- **Recommended way to build indices**

**Pipeline steps**:
1. Clone/update httpx → ~30s
2. Chunk code → ~10s
3. Build vector index → ~3min
4. Build BM25 index → ~10s
**Total: ~5 minutes**

---

### 🛠️ Utility Scripts

#### **visualize_graph.py** - Agent Visualization
```bash
python scripts/utilities/visualize_graph.py
```
- Generates Mermaid diagram of LangGraph
- Saves to `documents/agent_graph.mmd`
- View at https://mermaid.live/

---

## Key Features

### 1. **AST-Based Chunking**
Unlike simple line-based splitting, the chunking uses Python's AST to:
- Preserve function/class boundaries
- Extract docstrings automatically
- Maintain parent context (e.g., `Client.get`)
- Include relevant imports

### 2. **Hybrid Indexing**
Two complementary indices:
- **Vector (ChromaDB)**: Semantic similarity search
- **BM25**: Keyword-based retrieval
- Combined in hybrid search for best results

### 3. **Production Ready**
- Progress bars (tqdm)
- Comprehensive logging
- Error handling
- Configurable via `settings.py`
- Modular design (run individually or as pipeline)

---

## Usage Examples

### Quick Start (Recommended)
```bash
# Build everything
python scripts/data_pipeline/build_all_indices.py

# Start the agent
python app.py
```

### Rebuild Specific Index
```bash
# Update repository and rebuild vector index only
python scripts/data_pipeline/ingestion/clone_httpx_repo.py
python scripts/data_pipeline/indexing/build_vector_index.py
```

### Development/Testing
```bash
# Test chunking on updated code
python scripts/data_pipeline/processing/chunk_code.py

# Check output
cat data/processed/code_chunks.json | jq '.[:3]'
```

---

## Configuration

All scripts use `settings.py`:

```python
# Repository
HTTPX_REPO_DIR = Path("data/httpx")
REPO_EXCLUDE_PATTERNS = ["tests/*", "docs/*", "*.md", ...]

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Vector DB
CHROMA_PERSIST_DIRECTORY = "data/vector_db"
COLLECTION_NAME = "code_chunks"

# RAG Components
RAG_COMPONENTS_DIR = "data/rag_components"
```

---

## Output Files

After running the pipeline:

```
data/
├── httpx/                         # Git repository (gitignored)
│   ├── httpx/
│   ├── tests/
│   └── ...
│
├── processed/                     # Intermediate data (gitignored)
│   └── code_chunks.json          # ~500-1000 chunks
│
├── vector_db/                     # ChromaDB (gitignored)
│   └── chroma.sqlite3
│
└── rag_components/                # Pickle files (gitignored)
    ├── bm25_index.pkl            # BM25 index
    └── chunks.pkl                # Chunk data
```

---

## Comparison with RAG_agent

| Feature | RAG_agent | code_rag_agent |
|---------|-----------|----------------|
| **Input** | Documentation (Markdown) | Code (Python) |
| **Parsing** | Markdown parser | AST parser |
| **Chunking** | Section-based | Function/class-based |
| **Metadata** | Framework, topic | File path, line range |
| **Granularity** | 3-level (doc/section/block) | 1-level (code chunk) |
| **Output** | `output/json_files/` | `processed/code_chunks.json` |
| **Pipeline** | Download → Parse → Index | Clone → Chunk → Index |

---

## Next Steps

1. **Build indices**: `python scripts/data_pipeline/build_all_indices.py`
2. **Test agent**: `python app.py`
3. **Visualize graph**: `python scripts/utilities/visualize_graph.py`
4. **Rebuild on updates**: Re-run pipeline when httpx updates

---

## Documentation

See `scripts/README.md` for complete documentation including:
- Detailed usage instructions
- Troubleshooting guide
- Performance benchmarks
- Development guidelines
