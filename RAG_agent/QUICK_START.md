# Quick Start Guide

Get up and running with the LangGraph Documentation Assistant in 5 minutes.

## Prerequisites

- **Python**: 3.10 or higher (3.11+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~3GB total (includes models and documentation)
- **Internet**: Required for API keys, model downloads, and online mode

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag_agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Time**: ~2-3 minutes

### 3. Get API Keys

#### Google Gemini (Required)

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key

#### Tavily (Optional - Online Mode)

1. Visit [Tavily](https://tavily.com)
2. Sign up for free account (1000 searches/month)
3. Get API key from dashboard

### 4. Configure Environment

Create `.env` file:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (for online mode with web search)
AGENT_MODE=offline
TAVILY_API_KEY=your_tavily_key_here
```

### 5. Download Documentation & Build Indices

**Option A: Automated (Recommended)**

```bash
python init_project.py
```

This will:
- Download documentation (~10MB)
- Download ML models (~170MB)
- Build vector database (~45MB)
- Build search indices (~25MB)
- Validate everything works

**Time**: 5-10 minutes

**Option B: Manual**

```bash
# Download documentation
python scripts/download_docs.py

# Build vector database
python scripts/data_pipeline/indexing/build_vector_index.py

# Build search components
python scripts/data_pipeline/indexing/build_rag_components.py
```

### 6. Run the Chatbot

```bash
python chatbot.py
```

## Your First Query

```
You: How do I add persistence to a LangGraph agent?

Agent: To add persistence to a LangGraph agent, you need to use a checkpointer...

[Response with code examples and citations]

Sources:
- langgraph/python/concepts/persistence.json
- langgraph/python/how-tos/persistence-tutorial.json
```

## Operating Modes

### Offline Mode (Default)

Works without internet (except for Gemini API):

```bash
# In .env file
AGENT_MODE=offline
```

**Uses**:
- Local documentation
- Pre-built vector database
- BM25 keyword search

### Online Mode

Adds real-time web search:

```bash
# In .env file
AGENT_MODE=online
TAVILY_API_KEY=your_key_here
```

**Adds**:
- Latest documentation updates
- Community tutorials
- Recent feature announcements

## Common Commands

### Check Installation

```bash
# Verify dependencies
pip list | grep -E "langchain|chromadb|sentence-transformers"

# Check vector database
python -c "import chromadb; c=chromadb.PersistentClient('./data/vector_db'); print('Docs:', c.get_collection('documentation_embeddings').count())"
```

### Update Documentation

```bash
# Download latest docs
python scripts/download_docs.py

# Rebuild indices
python scripts/data_pipeline/indexing/build_rag_components.py
```

### Run Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### "No module named 'X'"

```bash
pip install -r requirements.txt --force-reinstall
```

### "GOOGLE_API_KEY not configured"

Make sure `.env` file exists and contains:
```
GOOGLE_API_KEY=your_actual_key_here
```

### "Vector database not found"

```bash
python scripts/data_pipeline/indexing/build_vector_index.py
```

### High Memory Usage

```bash
# Force CPU usage (disable GPU)
export CUDA_VISIBLE_DEVICES=""
python chatbot.py
```

## Next Steps

- Read [README.md](README.md) for comprehensive documentation
- Check [ARCHITECTURE.md](documents/ARCHITECTURE.md) for system design
- See [DEV_GUIDE.md](documents/DEV_GUIDE.md) for development help
- Review example queries in README

## Support

- Open an issue on GitHub
- Check troubleshooting section in README
- Review initialization logs: `init_project.log`

---

**Ready to build?** Start with: `python chatbot.py`
