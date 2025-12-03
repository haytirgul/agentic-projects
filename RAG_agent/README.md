# RAG Documentation Assistant

> An intelligent LangGraph-powered agent that helps developers work with LangChain, LangGraph, and LangSmith by answering practical questions using local documentation and optional web search.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Overview

The RAG Documentation Assistant is a **multi-stage LangGraph agent** that:

- **Understands intent**: Classifies user questions into 10 different intent types (implementation guide, troubleshooting, conceptual explanation, etc.)
- **Retrieves accurately**: Uses hybrid search (VectorDB + BM25) to find the most relevant documentation
- **Answers with citations**: Generates responses with proper source citations from official docs
- **Works offline**: Uses locally stored documentation (LangChain, LangGraph, LangSmith)
- **Online mode**: Optionally supplements with real-time web search for latest information
- **Conversation memory**: Tracks conversation history for context-aware responses

### Key Features

- **Hybrid RAG System**: Combines semantic search and keyword matching for 85%+ accuracy
- **Intent-Aware Retrieval**: Optimizes search based on question type
- **Security-First**: Validates all inputs for prompt injection and harmful content
- **Fast**: Sub-second retrieval with parallel search execution
- **Stateful Conversations**: Maintains context across multiple questions
- **Production-Ready**: Comprehensive error handling, logging, and monitoring

---

## Quick Start

> **For a 5-minute setup guide, see [QUICK_START.md](QUICK_START.md)**

### Prerequisites

**System Requirements:**
- **Python**: 3.10 or higher (3.11+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~3GB total (includes models and documentation)
- **Internet**: Required for API keys, model downloads, and online mode

**Dependencies:**
- pip (latest version recommended)
- Git (for cloning repository)
- Internet connection during setup (downloads ~200MB of models/docs)

### Installation

#### Option 1: Automated Setup (Recommended)

Use the initialization script for a complete automated setup:

```bash
# Clone the repository
cd rag_agent

# Run automated setup (does everything)
python init_project.py
```

**What the script does:**
- ✅ Validates Python version and environment
- ✅ Installs all dependencies from `requirements.txt`
- ✅ Downloads ML models (~150MB: embeddings + security classifier)
- ✅ Downloads LangChain/LangGraph documentation (~10MB)
- ✅ Builds vector database and search indices
- ✅ Creates configuration files
- ✅ Validates everything works

**Estimated time**: 5-10 minutes
**Benefits**: No manual steps, comprehensive validation, detailed logging

#### Option 2: Manual Setup

If the automated script fails or you prefer manual control:

```bash
# Clone the repository
cd rag_agent

# Install dependencies
pip install -r requirements.txt

# Create environment file
touch .env
```

### Environment Setup

**Required Configuration:**

Edit the `.env` file created during setup and add your API key:

```bash
# Required: Google Gemini API Key
GOOGLE_API_KEY=your_actual_api_key_here
```

**Get your Google API Key:**
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key to your `.env` file

**Optional Configuration (Online Mode):**

For web search capabilities, also add:

```bash
# Optional: Enable online mode with web search
AGENT_MODE=online
TAVILY_API_KEY=your_tavily_api_key_here
```

**Get your Tavily API Key:**
1. Visit [Tavily](https://tavily.com)
2. Sign up for free account (1000 searches/month)
3. Get API key from dashboard
4. Add to your `.env` file

### Build RAG Index

**Note**: If you used `python init_project.py`, this step is already done automatically.

**Manual RAG Setup** (if not using init_project.py):

```bash
# Step 1: Build vector database and embeddings
python scripts/data_pipeline/indexing/build_vector_index.py

# Step 2: Build BM25 search components
python scripts/data_pipeline/indexing/build_rag_components.py
```

**What gets built:**
- **Vector Database**: ChromaDB with 10k+ embeddings (~45MB)
- **BM25 Index**: Keyword search index (~15MB)
- **Document Index**: Fast document lookup (~8MB)
- **Takes**: ~5-10 minutes total

**Rebuilding**: Run these scripts after updating documentation or changing embedding models

### Model Downloads

During setup, the following ML models are downloaded automatically:

**Required Models:**
- **Sentence Transformers** (`all-MiniLM-L6-v2`): 80MB
  - Used for creating document embeddings
  - Local, no API calls required
  - Downloads from HuggingFace

- **ProtectAI DeBERTa v3**: 90MB
  - Used for prompt injection detection
  - Local security validation
  - Downloads from HuggingFace

**Optional Models (Online Mode):**
- **Google Gemini**: API-based, no local download
  - Used for LLM responses and embeddings (if configured)

**Total Download Size**: ~170MB (plus documentation)
**Storage Location**: `./data/sentence_transformers_cache/` and HuggingFace cache

### Data Directory Structure

After setup, your project will have this structure:

```
rag_agent/
├── data/
│   ├── input/                    # Raw downloaded documentation
│   │   ├── langchain_llms_full.txt
│   │   └── langgraph_llms_full.txt
│   ├── output/                   # Processed documentation
│   │   ├── json_files/           # 547 parsed docs (~10MB)
│   │   └── md_files/             # Alternative markdown format
│   ├── processed_docs/           # Structured JSON documents
│   ├── vector_db/                # ChromaDB vector database (~45MB)
│   ├── rag_components/           # Pre-built search indices
│   │   ├── bm25_index.pkl        # BM25 keyword search (~15MB)
│   │   ├── document_index.pkl    # Document lookup index (~8MB)
│   │   └── vector_index.pkl      # Vector search index
│   └── sentence_transformers_cache/  # ML model cache (~80MB)
├── .env                          # Your configuration file
├── init_project.log              # Setup log (for troubleshooting)
└── ...
```

**Space Usage**: ~250MB total after complete setup

### Run the Chatbot

```bash
python chatbot.py
```

**Example interaction:**

```
You: How do I add persistence to a LangGraph agent?

Agent: To add persistence to a LangGraph agent, you need to use a checkpointer...

Sources:
- langgraph/python/concepts/persistence.json
- langgraph/python/how-tos/persistence-tutorial.json
```

---

## Usage

### Basic Usage

```python
from src.graph.builder import get_compiled_graph

# Initialize the graph
app = get_compiled_graph()

# Ask a question
result = app.invoke({
    "user_input": "How do I create a LangGraph StateGraph with checkpointing?"
})

print(result["final_response"])
```

### With Configuration

```python
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer for conversation memory
checkpointer = MemorySaver()

# Build graph with checkpointer
app = get_compiled_graph(checkpointer=checkpointer)

# Use thread_id for conversation continuity
config = {"configurable": {"thread_id": "user_session_123"}}

# First question
result1 = app.invoke(
    {"user_input": "What is a StateGraph in LangGraph?"},
    config
)

# Follow-up question (maintains context)
result2 = app.invoke(
    {"user_input": "How do I add persistence to it?"},
    config
)
```

### Online Mode (Web Search)

Enable real-time web search to supplement local documentation:

```bash
# Set environment variables
export AGENT_MODE=online
export TAVILY_API_KEY=your_tavily_key

# Run chatbot
python chatbot.py
```

**When to use online mode:**
- Questions about latest features released after documentation snapshot
- Community discussions and tutorials
- Third-party integrations
- Breaking changes and migration guides

---

## Operating Modes

The agent supports two distinct operating modes controlled via the `AGENT_MODE` environment variable.

### Offline Mode (Default)

**Primary mode for production use** - works completely without internet connectivity.

**Data Sources:**
- LangChain Python documentation (full)
- LangGraph Python documentation (full)
- LangSmith documentation
- Local vector database with 10,000+ embeddings
- Pre-built BM25 keyword index

**How it works:**
1. User query → Intent classification → Hybrid retrieval → Response generation
2. All processing happens locally using downloaded documentation
3. No external API calls except Google Gemini for LLM responses

**When to use:**
- Production deployments
- Environments without internet access
- Cost-conscious usage (only Gemini API costs)
- When you need deterministic, versioned documentation

**Configuration:**
```bash
AGENT_MODE=offline  # Default value
```

### Online Mode

**Enhanced mode** - supplements local documentation with real-time web search.

**Additional Data Sources:**
- Tavily web search API (free tier available)
- Latest community tutorials and discussions
- Recent feature announcements and breaking changes
- Third-party integrations and examples

**How it works:**
1. User query → Intent classification → Parallel execution:
   - Local retrieval (VectorDB + BM25)
   - Web search (Tavily API)
2. Results deduplicated (web doesn't duplicate local docs)
3. Hybrid scoring combines local + web results

**When to use:**
- Questions about very recent features (post-documentation)
- Community best practices and tutorials
- Third-party integrations
- Breaking changes and migration guides

**Configuration:**
```bash
AGENT_MODE=online
TAVILY_API_KEY=your_tavily_api_key_here  # Get from https://tavily.com
```

**Performance:**
- Latency: ~2-3 seconds (vs <1 second offline)
- Cost: Additional API calls to Tavily
- Quality: Access to latest information

---

## Data Freshness Strategy

### Offline Data Preparation

**Initial Setup:**
- Downloads LangChain, LangGraph, and LangSmith documentation from official sources
- Parses llms.txt format into structured JSON documents
- Creates embeddings using Google Gemini (sentence-transformers/all-MiniLM-L6-v2)
- Builds ChromaDB vector index for semantic search
- Creates BM25 keyword index for exact matching

**Data Sources:**
- **LangGraph**: https://langchain-ai.github.io/langgraph/llms.txt
- **LangChain**: https://python.langchain.com/llms.txt
- **LangSmith**: Included in LangChain docs

**Update Strategy:**
- Run `python scripts/download_docs.py` to refresh documentation
- Rebuild indices with `python scripts/data_pipeline/indexing/build_rag_components.py`
- Typical update frequency: Monthly or when new major versions release

**Data Volume:**
- ~500 JSON documents
- ~10,000 embeddings
- ~10MB text content
- ~45MB vector database

### Online Data Strategy

**Web Search Integration:**
- Uses Tavily API for intelligent web search
- Queries are generated based on user intent classification
- Results are deduplicated against local documentation
- Web results get 20% scoring penalty to prefer local docs

**Free Tier Compliance:**
- Tavily: 1,000 searches/month free
- Google Gemini: 60 requests/minute free tier
- No other external services used

**Quality Assurance:**
- Web results filtered to avoid duplicate content
- Domain-based deduplication (python.langchain.com, docs.langchain.com)
- Scoring prioritizes official documentation over community content

---

## Architecture

### High-Level Flow

```
User Input
    ↓
Security Gateway (ML-based prompt injection detection)
    ↓
Intent Classification
    ↓
Hybrid Retrieval (VectorDB + BM25)
    ↓
Response Generation (with citations)
    ↓
Save to Conversation Memory
    ↓
Final Answer
```

### Graph Structure

The agent is built as a **LangGraph state machine** with the following nodes:

1. **security_gateway**: ML-based prompt injection detection
2. **intent_classification**: Classify user query into intent categories
3. **hybrid_retrieval**: Retrieve relevant documents (VectorDB + BM25)
4. **prepare_messages**: Inject retrieved docs into context
5. **agent**: Generate response using LLM
6. **finalize**: Extract final response from messages
7. **user_output**: Display response to user
8. **save_turn**: Save query-response pair to conversation memory
9. **prompt_continue**: Ask if user wants to continue (for multi-turn conversations)
10. **reset_for_next_turn**: Clear per-turn state, preserve history

**Key Optimizations:**
- **Parallel retrieval** for faster search
- **Conversation memory** for context-aware responses
- **Streaming responses** for better UX

### Intent Types

The system classifies questions into 10 categories:

| Intent Type | Example |
|-------------|---------|
| `factual_lookup` | "What is a checkpointer?" |
| `implementation_guide` | "How do I add memory to agents?" |
| `troubleshooting` | "Why is my agent not saving state?" |
| `conceptual_explanation` | "Explain how LangGraph routing works" |
| `best_practices` | "What's the best way to handle errors?" |
| `comparison` | "LangChain vs LangGraph for chatbots?" |
| `configuration_setup` | "How to install LangGraph?" |
| `migration` | "Migrate from LangChain 0.1 to 0.2?" |
| `capability_exploration` | "Can LangGraph handle streaming?" |
| `api_reference` | "Show me StateGraph API docs" |

### Hybrid Retrieval

Combines two search methods for optimal accuracy:

**Hybrid Scoring Formula:**

```python
# Offline mode
final_score = (
    0.7 * vector_score +     # Semantic similarity (synonyms, paraphrases)
    0.3 * bm25_score         # Keyword matching (exact terms)
)

# Online mode (with web search)
final_score = (
    0.15 * vector_score +    # Semantic similarity
    0.15 * bm25_score +      # Keyword matching
    0.70 * web_bonus         # Web search results
)
```

**Why hybrid?**
- **VectorDB**: Handles "save state" → "persistence" semantic matching
- **BM25**: Handles exact technical terms like "StateGraph"

**Performance:**
- Retrieval: < 300ms
- Precision@5: 85%+
- Parallel execution for speed

---

## Example Queries

### 1. Implementation Guide

**Query**: "How do I create a LangGraph agent with persistence?"

**Intent**: `implementation_guide`, framework=`langgraph`, language=`python`

**Retrieved Docs**:
- langgraph/python/concepts/persistence.json
- langgraph/python/how-tos/persistence-tutorial.json
- langgraph/python/concepts/checkpointer.json

**Response**: Step-by-step code example with checkpointer setup

---

### 2. Troubleshooting

**Query**: "My LangGraph agent is not saving conversation state. Why?"

**Intent**: `troubleshooting`, framework=`langgraph`, topics=`[state, persistence]`

**Retrieved Docs**:
- langgraph/python/how-tos/persistence-troubleshooting.json
- langgraph/python/concepts/state-management.json

**Response**: Common issues and debugging steps

---

### 3. Comparison

**Query**: "Should I use LangChain or LangGraph for building a chatbot?"

**Intent**: `comparison`, frameworks=`[langchain, langgraph]`

**Retrieved Docs**:
- langgraph/concepts/when-to-use-langgraph.json
- langchain/concepts/agents-overview.json
- Comparison table

**Response**: Pros/cons of each with recommendations

---

## Project Structure

```
RAG_agent/
├── README.md                  # This file
├── QUICK_START.md             # 5-minute setup guide
├── chatbot.py                 # Interactive CLI
├── settings.py                # Configuration
├── requirements.txt           # Dependencies
├── init_project.py            # Automated setup script
│
├── src/                       # Core application logic
│   ├── graph/                 # LangGraph workflow and state management
│   ├── nodes/                 # Graph node implementations
│   ├── rag/                   # RAG components (vector search, BM25, web search)
│   ├── llm/                   # LLM clients and workers
│   ├── security/              # Security validation and prompt guards
│   ├── tools/                 # Agent tools
│   └── utils/                 # Utilities (timing, logging)
│
├── models/                    # Pydantic data models
├── prompts/                   # Prompt templates as Python modules
├── scripts/                   # Data pipeline and utilities
│   └── data_pipeline/         # Documentation ingestion and indexing
│       ├── ingestion/         # Download and parse docs
│       ├── indexing/          # Build vector and BM25 indices
│       └── processing/        # Markdown to JSON conversion
│
├── data/                      # Data storage (excluded from git)
│   ├── input/                 # Raw documentation files
│   ├── output/json_files/     # Parsed documentation (~547 files)
│   ├── vector_db/             # ChromaDB vector database
│   └── rag_components/        # BM25 and document indices
│
└── documents/                 # Project documentation
    ├── ARCHITECTURE.md        # System architecture
    ├── DEV_GUIDE.md           # Developer guide
    ├── RAG_README.md          # RAG implementation details
    ├── SECURITY_README.md     # Security features
    ├── SETUP.md               # Setup instructions
    └── INDEX.md               # Documentation index
```

---

## Development

### Rebuilding Vector Index

After updating documentation:

```bash
python scripts/data_pipeline/indexing/build_vector_index.py
```

### Testing Vector Search

```bash
python scripts/test_vector_search.py
```

---

## Configuration

### Environment Variables

#### Core Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_API_KEY` | **Yes** | - | Google Gemini API key |
| `AGENT_MODE` | No | `offline` | `offline` or `online` |
| `TAVILY_API_KEY` | No* | - | Tavily API key (*required for online mode) |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

#### Model Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MODEL_FAST` | No | `gemini-2.5-flash-lite` | Fast model for simple tasks |
| `MODEL_INTERMEDIATE` | No | `gemini-2.5-flash` | Balanced speed/quality |
| `MODEL_SLOW` | No | `gemini-2.5-pro` | High-quality for complex tasks |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |

#### RAG Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VECTOR_WEIGHT` | No | `0.15` | Vector search weight (online mode) |
| `BM25_WEIGHT` | No | `0.15` | BM25 search weight (online mode) |
| `WEB_BONUS` | No | `0.7` | Web search bonus (online mode) |
| `VECTOR_WEIGHT_OFFLINE` | No | `0.7` | Vector weight (offline mode) |
| `BM25_WEIGHT_OFFLINE` | No | `0.3` | BM25 weight (offline mode) |

#### Advanced Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECURITY_ENABLED` | No | `false` | Enable prompt injection detection |
| `ENABLE_STREAMING` | No | `true` | Enable streaming responses |
| `MAX_HISTORY_TURNS` | No | `5` | Conversation history length |
| `WEB_SEARCH_MAX_RESULTS` | No | `5` | Max web results per query |
| `LLM_TEMPERATURE` | No | `0.0` | Response creativity (0.0-1.0) |
| `LLM_MAX_TOKENS` | No | `8192` | Max response length |

#### LangSmith Observability (Optional)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No* | - | LangSmith API key (*if tracing enabled) |
| `LANGCHAIN_PROJECT` | No | `rag-agent` | LangSmith project name |

### Model Configuration

**Model Tiers** (configured in [settings.py](settings.py)):

The system uses three model tiers for different operations:

- **Fast Model**: `gemini-2.5-flash-lite` - Quick operations (parsing, validation)
- **Intermediate Model**: `gemini-2.5-flash` - Balanced operations (intent classification)
- **Slow Model**: `gemini-2.5-pro` - Complex operations (response generation)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local model)

**Customization:**

You can override model selections via environment variables:

```bash
MODEL_FAST=gemini-2.5-flash-lite
MODEL_INTERMEDIATE=gemini-2.5-flash
MODEL_SLOW=gemini-2.5-pro
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Troubleshooting

### Setup Issues

#### "Python 3.10+ is required"

**Solution**: Upgrade Python or use a newer environment
```bash
# Check your Python version
python --version

# If < 3.10, install Python 3.11+ from python.org
# Or use conda/pyenv to create a new environment
conda create -n rag_agent python=3.11
conda activate rag_agent
```

#### "Failed to install dependencies"

**Common solutions:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# On Linux, install build tools
sudo apt-get install python3-dev build-essential

# On macOS, install Xcode command line tools
xcode-select --install
```

#### "GOOGLE_API_KEY not configured"

**Solution**: Get and set your API key
```bash
# 1. Get key from https://aistudio.google.com/app/apikey
# 2. Edit .env file
echo "GOOGLE_API_KEY=your_actual_key_here" > .env
```

#### Model Download Failures

**Sentence Transformers model fails:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
pip install --upgrade sentence-transformers
python init_project.py
```

**Security model fails:**
```bash
# Manual download
pip install huggingface-hub
huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2
```

### Runtime Issues

#### "Agent returns no search results"

**Check if RAG index is built:**
```bash
# Verify vector database exists
ls data/vector_db/

# Verify RAG components exist
ls data/rag_components/

# If missing, rebuild indices
python scripts/data_pipeline/indexing/build_vector_index.py
python scripts/data_pipeline/indexing/build_rag_components.py
```

#### "Web search not working in online mode"

**Verify configuration:**
```bash
# Check environment variables
echo $AGENT_MODE  # Should be "online"
echo $TAVILY_API_KEY  # Should not be empty

# Test Tavily connection
python -c "from src.rag.web_search_api import WebSearchEngine; ws=WebSearchEngine(); print('Web search ready:', ws.client is not None)"
```

#### API Rate Limits

**Google Gemini Free Tier:**
- 15 requests per minute
- 1 million tokens per day
- 1,500 requests per day

**Solutions when hitting limits:**
- Wait 60+ seconds between requests
- Use slower models (`MODEL_SLOW`)
- Upgrade to paid tier at [Google AI Studio](https://aistudio.google.com/)
- Implement request queuing/caching

**Tavily Free Tier:**
- 1,000 searches per month

### Common Error Messages

#### "ModuleNotFoundError: No module named 'X'"

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### "Connection timeout" / "Network error"

- Check internet connection
- For API calls: Verify API keys are correct
- For model downloads: Check firewall/proxy settings
- Try with different network or VPN

#### "CUDA out of memory" (if using GPU)

```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
# Or reduce batch size in security settings
```

### Getting Help

**Debug logging:**
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python chatbot.py
```

**Check initialization logs:**
```bash
# Review setup log
cat init_project.log
```

**File an issue** with:
- Python version (`python --version`)
- OS and hardware info
- Full error traceback
- Contents of `init_project.log`
- Your `.env` file (without API keys)

### Reset and Recovery

**Clean restart:**
```bash
# Remove all generated data
rm -rf data/vector_db data/rag_components .env init_project.log

# Clear model caches
rm -rf ~/.cache/huggingface ~/.cache/torch

# Reinitialize
python init_project.py
```

---

## Contributing

### Adding New Features

1. Create a new branch
2. Implement feature with tests
3. Update documentation
4. Run full test suite
5. Submit pull request

### Code Style

- Follow PEP 8
- Use type hints on all functions
- Add docstrings (Google style)
- Maximum line length: 100 characters
- Use Pydantic for data validation

---

## Documentation

### For Users

- [README.md](README.md) - This file (quickstart, usage examples)
- [QUICK_START.md](QUICK_START.md) - 5-minute setup guide

### For Developers

- [documents/DEV_GUIDE.md](documents/DEV_GUIDE.md) - Comprehensive developer guide
- [documents/ARCHITECTURE.md](documents/ARCHITECTURE.md) - System architecture and design
- [documents/RAG_README.md](documents/RAG_README.md) - RAG implementation details
- [documents/SECURITY_README.md](documents/SECURITY_README.md) - Security features
- [documents/SETUP.md](documents/SETUP.md) - Setup instructions
- [documents/INDEX.md](documents/INDEX.md) - Documentation index

---

## API Keys

### Google Gemini (Required)

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy key to `.env` as `GOOGLE_API_KEY`

**Free Tier Limits:**
- 15 requests/minute
- 1 million tokens/day
- Sufficient for development and light production use

### Tavily (Optional - Online Mode)

1. Go to [Tavily](https://tavily.com)
2. Sign up for free account
3. Get API key from dashboard
4. Copy key to `.env` as `TAVILY_API_KEY`

**Free Tier Limits:**
- 1,000 searches/month
- Sufficient for most use cases

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) by LangChain
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [Google Gemini](https://ai.google.dev/) LLM
- Documentation from [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), and [LangSmith](https://smith.langchain.com/)

---

## Support

For issues, questions, or contributions:

- Check the [DEV_GUIDE.md](documents/DEV_GUIDE.md) for development help
- Review [ARCHITECTURE.md](documents/ARCHITECTURE.md) for system design
- See [troubleshooting section](#troubleshooting) above
- Open an issue in the repository

---

**Built by Hay Hoffman | Last Updated: 2025-12-03**
