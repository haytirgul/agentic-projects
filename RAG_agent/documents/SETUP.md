# Setup Guide

Complete setup instructions for first-time installation of rag_agent.

## Prerequisites

- **Python**: 3.10 or higher
- **Git**: For cloning the repository
- **Internet**: For downloading models and documentation (~500MB total)
- **Disk Space**: ~2GB for models, documentation, and indices

## Quick Start (Recommended)

### 1. Clone Repository

```bash
git clone <repository-url>
cd rag_agent
```

### 2. Run Initialization Script

```bash
python init_project.py
```

This script will:
- ✓ Validate Python version
- ✓ Install all dependencies from `requirements.txt`
- ✓ Create necessary directories
- ✓ Check/create `.env` configuration file
- ✓ Download security classifier (ProtectAI DeBERTa v3 - ~90MB)
- ✓ Download embedding model (all-MiniLM-L6-v2 - ~80MB)
- ✓ Download LangChain/LangGraph documentation (~10MB)
- ✓ Process documentation (parse llms.txt format)
- ✓ Build ChromaDB vector database (~50MB)
- ✓ Build BM25 search index (~20MB)

**Estimated time**: 5-10 minutes (depending on internet speed)

### 3. Configure API Keys

Edit `.env` file and add your API key:

```bash
GOOGLE_API_KEY=your_actual_api_key_here
```

### 4. Run Chatbot

```bash
python chatbot.py
```

## Manual Setup (Alternative)

If the initialization script fails or you prefer manual setup:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create Directories

```bash
mkdir -p data/raw_docs data/processed_docs data/chroma_db data/outputs
```

### Step 3: Configure Environment

Create `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
LOG_LEVEL=INFO
INTENT_MODEL=gemini-2.0-flash-exp
AGENT_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Step 4: Download Models

```python
# Security model
python -c "from src.security import PromptGuardClassifier; PromptGuardClassifier()"

# Embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Step 5: Build RAG Index

```bash
python data_processing/build_rag_index.py
```

This will:
- Download LangChain/LangGraph docs
- Parse and preprocess
- Build vector database (ChromaDB)
- Build BM25 index

**Time**: 5-10 minutes

### Step 6: Test Installation

```bash
python chatbot.py
```

## Initialization Scripts

### `init_project.py` - Full Setup

Comprehensive initialization with validation:

```bash
python init_project.py
```

**Features:**
- ✓ Step-by-step progress tracking
- ✓ Validates each component
- ✓ Detailed logging to `init_project.log`
- ✓ Error handling and recovery
- ✓ Summary report at end

**Use when**: First-time setup or after repo updates

### `quick_init.py` - Minimal Setup

Fast setup for testing (without RAG):

```bash
python quick_init.py
```

**Features:**
- ✓ Installs dependencies only
- ✓ Downloads models only
- ✓ Creates `.env` template
- ✗ Skips documentation download
- ✗ Skips RAG index building

**Use when**: Testing security/validation features only

## Troubleshooting

### Python Version Error

```
Error: Python 3.10+ is required
```

**Solution**: Upgrade Python:
```bash
python --version  # Check current version
# Install Python 3.10+ from python.org
```

### pip Install Fails

```
Error: Failed to install dependencies
```

**Solutions**:
1. Upgrade pip: `python -m pip install --upgrade pip`
2. Install build tools (if on Linux): `sudo apt-get install python3-dev`
3. Try with `--no-cache-dir`: `pip install --no-cache-dir -r requirements.txt`

### GOOGLE_API_KEY Not Set

```
Warning: GOOGLE_API_KEY not set in .env file
```

**Solution**:
1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Edit `.env` file:
   ```env
   GOOGLE_API_KEY=AIza...your_key_here
   ```

### Security Model Download Fails

```
Error: Failed to initialize security classifier
```

**Solutions**:
1. Check internet connection
2. Try manual download:
   ```bash
   pip install huggingface-hub
   huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2
   ```
3. Check HuggingFace is accessible (no firewall blocking)

### Vector Database Empty

```
Error: Vector database is empty
```

**Solution**: Rebuild RAG index:
```bash
rm -rf data/chroma_db data/bm25_index.pkl
python data_processing/build_rag_index.py
```

### Documentation Download Fails

```
Error: Failed to download documentation
```

**Solutions**:
1. Check internet connection
2. Check GitHub access (may need VPN if restricted)
3. Manual download:
   ```bash
   cd data_processing
   python langchain_parser.py
   python langgraph_parser.py
   ```

## Directory Structure After Setup

```
rag_agent/
├── data/
│   ├── raw_docs/              # Downloaded documentation
│   │   ├── langchain_docs_llms_txt.md
│   │   └── langgraph_docs_llms_txt.md
│   ├── processed_docs/        # Parsed JSON documents
│   │   ├── langchain_docs_processed.json
│   │   └── langgraph_docs_processed.json
│   ├── chroma_db/             # Vector database (~50MB)
│   ├── bm25_index.pkl         # BM25 search index (~20MB)
│   └── outputs/               # Agent outputs
├── src/
│   ├── security/              # Security validation
│   ├── nodes/                 # LangGraph nodes
│   ├── tools/                 # Agent tools
│   └── ...
├── .env                       # Environment configuration
├── init_project.py            # Full initialization script
├── quick_init.py              # Quick setup script
├── chatbot.py                 # Main chatbot application
└── init_project.log           # Initialization log
```

## Verification

After setup, verify everything works:

### 1. Check Models

```python
# Security model
python -c "from src.security import PromptGuardClassifier; c=PromptGuardClassifier(); print('Security:', c.is_available)"

# Embedding model
python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Embeddings: OK')"
```

### 2. Check Vector Database

```python
python -c "import chromadb; c=chromadb.PersistentClient('data/chroma_db'); print('Docs:', c.get_collection('langchain_langgraph_docs').count())"
```

Expected: `Docs: 500+` (number varies)

### 3. Check BM25 Index

```python
python -c "import pickle; d=pickle.load(open('data/bm25_index.pkl','rb')); print('BM25 docs:', len(d['documents']))"
```

Expected: `BM25 docs: 500+`

### 4. Run Chatbot

```bash
python chatbot.py
```

Expected: Chatbot starts and prompts for input

## Environment Variables

Required variables in `.env`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | ✓ Yes | - | Google Gemini API key |
| `LOG_LEVEL` | No | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `INTENT_MODEL` | No | gemini-2.0-flash-exp | Model for intent classification |
| `AGENT_MODEL` | No | gemini-2.0-flash-exp | Model for agent responses |
| `EMBEDDING_MODEL` | No | sentence-transformers/all-MiniLM-L6-v2 | Embedding model |
| `VECTOR_DB_COLLECTION` | No | langchain_langgraph_docs | ChromaDB collection name |
| `TOP_K_RETRIEVAL` | No | 25 | Initial retrieval count |
| `FINAL_TOP_K` | No | 5 | Final top-K after reranking |
| `ENABLE_STREAMING` | No | false | Enable streaming responses |
| `AGENT_TEMPERATURE` | No | 0.0 | LLM temperature |
| `AGENT_MAX_TOKENS` | No | 4096 | Max response tokens |

## Updates and Re-initialization

### Update Documentation

```bash
# Delete existing data
rm -rf data/raw_docs data/processed_docs data/chroma_db data/bm25_index.pkl

# Rebuild
python data_processing/build_rag_index.py
```

### Update Models

```bash
# Clear model cache
rm -rf ~/.cache/huggingface
rm -rf ~/.cache/torch

# Re-download
python init_project.py
```

### Reset Everything

```bash
# Nuclear option: delete all data and models
rm -rf data .env init_project.log
rm -rf ~/.cache/huggingface ~/.cache/torch

# Full re-initialization
python init_project.py
```

## Getting Help

1. **Check logs**: `init_project.log` has detailed information
2. **Check documentation**: See `documents/` directory for architecture details
3. **File an issue**: Include logs and error messages

## Next Steps

After successful setup:

1. **Test the chatbot**: `python chatbot.py`
2. **Try example queries**:
   - "How do I add persistence to LangGraph?"
   - "What are the different LangChain agents?"
   - "How do I use tools in LangGraph?"
3. **Read documentation**:
   - [Security Gateway](documents/SECURITY_GATEWAY.md)
   - [Architecture Overview](documents/)
4. **Customize configuration**: Edit `.env` to use different models

---

**Last Updated**: 2025-11-30
**Version**: 1.0.0
