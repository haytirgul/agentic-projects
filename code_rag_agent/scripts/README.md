# Scripts - Code RAG Agent

This directory contains scripts for building the data pipeline and utilities for the code RAG agent.

## Directory Structure

```
scripts/
├── data_pipeline/              # Data processing pipeline
│   ├── ingestion/             # Repository cloning
│   │   └── clone_httpx_repo.py
│   ├── processing/            # Code chunking
│   │   └── chunk_code.py
│   ├── indexing/              # Index building
│   │   ├── build_vector_index.py
│   │   └── build_bm25_index.py
│   └── build_all_indices.py   # Master pipeline script
│
└── utilities/                  # Utility scripts
    └── visualize_graph.py     # Graph visualization
```

## Quick Start

### Build All Indices (Recommended)

Run the complete pipeline in one command:

```bash
python scripts/data_pipeline/build_all_indices.py
```

This will:
1. Clone/update the httpx repository
2. Parse and chunk Python code files
3. Build ChromaDB vector index
4. Build BM25 keyword index

**Expected time**: 5-10 minutes

---

## Individual Scripts

### 1. Ingestion

#### Clone httpx Repository

```bash
python scripts/data_pipeline/ingestion/clone_httpx_repo.py
```

**What it does**:
- Clones `https://github.com/encode/httpx.git` to `data/httpx/`
- If repo exists, pulls latest changes
- Shows commit info and Python file count

**Output**: `data/httpx/` (gitignored)

---

### 2. Processing

#### Chunk Code Files

```bash
python scripts/data_pipeline/processing/chunk_code.py
```

**What it does**:
- Walks through httpx repository
- Parses Python files using AST
- Extracts functions, classes, and methods
- Creates semantic code chunks with metadata

**Input**: `data/httpx/` (httpx repository)
**Output**: `data/processed/code_chunks.json`

**Chunk structure**:
```json
{
  "file_path": "httpx/_client.py",
  "start_line": 42,
  "end_line": 67,
  "chunk_type": "function",
  "name": "get",
  "parent_context": "Client",
  "full_name": "Client.get",
  "content": "def get(self, url, ...):\n    ...",
  "docstring": "Send a GET request.",
  "imports": ["import httpx", "from typing import ..."]
}
```

---

### 3. Indexing

#### Build Vector Index

```bash
python scripts/data_pipeline/indexing/build_vector_index.py
```

**What it does**:
- Loads code chunks from JSON
- Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Stores in ChromaDB for semantic search

**Input**: `data/processed/code_chunks.json`
**Output**: `data/vector_db/` (ChromaDB persistent storage)

**Configuration** (in `settings.py`):
- `EMBEDDING_MODEL`: Embedding model name
- `EMBEDDING_BATCH_SIZE`: Batch size for encoding (default: 32)
- `COLLECTION_NAME`: ChromaDB collection name

---

#### Build BM25 Index

```bash
python scripts/data_pipeline/indexing/build_bm25_index.py
```

**What it does**:
- Loads code chunks from JSON
- Tokenizes code content
- Builds BM25 index for keyword search
- Saves as pickle file

**Input**: `data/processed/code_chunks.json`
**Output**:
- `data/rag_components/bm25_index.pkl` (BM25 index)
- `data/rag_components/chunks.pkl` (Chunk data)

---

## Utilities

### Visualize Agent Graph

```bash
python scripts/utilities/visualize_graph.py
```

**What it does**:
- Loads the LangGraph agent definition
- Generates Mermaid diagram
- Saves to `documents/agent_graph.mmd`

**To view**:
1. Open https://mermaid.live/
2. Paste contents of `agent_graph.mmd`
3. View interactive diagram

---

## Pipeline Output

After running `build_all_indices.py`, you'll have:

```
data/
├── httpx/                      # Cloned repository (gitignored)
├── processed/
│   └── code_chunks.json       # Parsed code chunks
├── vector_db/                 # ChromaDB storage (gitignored)
└── rag_components/            # Pickle files (gitignored)
    ├── bm25_index.pkl
    └── chunks.pkl
```

---

## Troubleshooting

### Git not found
**Error**: `git: command not found`

**Solution**: Install git or add to PATH

### ChromaDB errors
**Error**: `chromadb` module not found

**Solution**:
```bash
pip install -r requirements.txt
```

### Out of memory
**Error**: Crash during embedding generation

**Solution**: Reduce `EMBEDDING_BATCH_SIZE` in `settings.py`

### Empty chunks
**Error**: `Created 0 code chunks`

**Solution**:
1. Check `HTTPX_REPO_DIR` exists and has Python files
2. Review `REPO_EXCLUDE_PATTERNS` in `settings.py`
3. Check file permissions

---

## Configuration

All scripts use settings from `settings.py`:

```python
# Repository
HTTPX_REPO_DIR = "data/httpx"
REPO_EXCLUDE_PATTERNS = ["tests/*", "docs/*", "*.md", ...]

# Chunking
CHUNK_STRATEGY = "ast"  # AST-based chunking

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32

# Vector DB
CHROMA_PERSIST_DIRECTORY = "data/vector_db"
COLLECTION_NAME = "code_chunks"
```

---

## Performance

**Typical pipeline times** (on httpx ~15k LOC):

| Step | Time | Output |
|------|------|--------|
| Clone repo | 10-30s | ~15k LOC |
| Chunk code | 5-15s | ~500-1000 chunks |
| Build vector index | 2-5min | ChromaDB with embeddings |
| Build BM25 index | 5-10s | BM25 pickle file |
| **Total** | **5-10min** | Ready for queries |

---

## Next Steps

After building indices:

1. **Start the agent**:
   ```bash
   python app.py
   ```

2. **Ask questions**:
   - "How does httpx validate SSL certificates?"
   - "Where is proxy support implemented?"
   - "What happens when a request times out?"

3. **Rebuild indices** (when httpx updates):
   ```bash
   python scripts/data_pipeline/build_all_indices.py
   ```

---

## Development

### Adding new scripts

1. Create script in appropriate directory
2. Follow naming convention: `verb_noun.py` (e.g., `build_index.py`)
3. Add docstring with usage instructions
4. Add to `build_all_indices.py` if part of pipeline

### Testing individual components

Each script can be run independently for testing:

```bash
# Test chunking only
python scripts/data_pipeline/processing/chunk_code.py

# Test vector index only
python scripts/data_pipeline/indexing/build_vector_index.py
```

---

## References

- **ChromaDB**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **BM25**: https://en.wikipedia.org/wiki/Okapi_BM25
- **LangGraph**: https://langchain-ai.github.io/langgraph/
