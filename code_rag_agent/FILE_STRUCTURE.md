# Code RAG Agent - File Structure

## ✅ Complete Directory Structure

```
code_rag_agent/
├── .env.example              # Environment variable template
├── .gitignore               # Git ignore patterns
├── settings.py              # Centralized configuration (adapted)
├── requirements.txt         # Python dependencies (adapted)
├── app.py                   # Interactive CLI (copied from chatbot.py)
├── SETUP_STATUS.md          # Detailed setup and next steps guide
├── FILE_STRUCTURE.md        # This file
│
├── documents/
│   └── task_prd.md          # Product Requirements Document (updated with conversation memory)
│
├── data/
│   ├── httpx/               # httpx repository clone (gitignored)
│   ├── index/               # Generated indices (gitignored)
│   ├── vector_db/           # ChromaDB storage (gitignored)
│   └── rag_components/      # Pickle files (gitignored)
│
├── models/
│   ├── __init__.py
│   └── conversation.py      # ✅ ConversationMemory & ConversationTurn
│
├── src/
│   ├── __init__.py
│   │
│   ├── indexing/            # ⚠️ TO BE CREATED
│   │   ├── __init__.py
│   │   ├── chunker.py       # (TODO) AST-based code chunking
│   │   ├── embedder.py      # (TODO) Embedding generation
│   │   └── index_builder.py # (TODO) Build ChromaDB index
│   │
│   ├── retrieval/           # ⚠️ TO BE CREATED
│   │   ├── __init__.py
│   │   └── reranker.py      # (TODO) Reranking with metrics
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py         # ✅ LangGraph builder (needs adaptation)
│   │   ├── state.py         # ✅ Agent state (needs adaptation)
│   │   ├── routing.py       # ✅ Routing logic (needs simplification)
│   │   │
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── conversation_memory.py  # ✅ Memory tracking
│   │       ├── retriever.py            # ✅ Hybrid retrieval (needs adaptation)
│   │       ├── router.py               # (TODO) Query analysis
│   │       └── synthesizer.py          # (TODO) Answer generation
│   │
│   ├── synthesis/           # ⚠️ TO BE CREATED
│   │   ├── __init__.py
│   │   ├── citation_extractor.py  # (TODO) Extract file:line citations
│   │   └── answer_formatter.py    # (TODO) Format answers
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── vector_search.py    # ✅ FAISS vector search
│   │   ├── bm25_search.py      # ✅ BM25 keyword search (needs adaptation)
│   │   ├── hybrid_scorer.py    # ✅ Hybrid scoring + RRF reranking
│   │   └── document_index.py   # ✅ In-memory index (needs renaming)
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── llm.py          # ✅ LLM factory (Gemini)
│   │   └── workers.py      # ✅ Structured output parsing
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py       # ✅ Logging utilities
│
├── prompts/                 # ⚠️ TO BE CREATED
│   ├── __init__.py
│   ├── router_prompts.py        # (TODO) Router node prompts
│   ├── retriever_prompts.py     # (TODO) Retriever prompts
│   └── synthesizer_prompts.py   # (TODO) Synthesizer prompts
│
└── tests/                   # ⚠️ TO BE CREATED
    ├── __init__.py
    ├── test_chunker.py      # (TODO) Chunking tests
    ├── test_retrieval.py    # (TODO) Retrieval tests
    └── test_agent.py        # (TODO) End-to-end tests
```

## Legend

- ✅ **Copied and ready** - File exists and is functional (may need adaptation)
- ⚠️ **Needs creation** - Directory exists but files need to be created
- **(needs adaptation)** - File copied but needs modification for code RAG
- **(TODO)** - File doesn't exist yet, needs to be created

## Summary

### ✅ Fully Complete (No Changes Needed)
- Directory structure
- `models/conversation.py` - Conversation memory
- `src/llm/llm.py` - LLM factory
- `src/llm/workers.py` - Structured output
- `src/rag/hybrid_scorer.py` - Hybrid scoring
- `src/utils/logger.py` - Logging
- `settings.py` - Configuration
- `requirements.txt` - Dependencies
- `.gitignore` - Git ignore
- `.env.example` - Environment template

### ⚠️ Copied but Needs Adaptation
- `src/rag/vector_search.py` - Change from docs to code chunks
- `src/rag/bm25_search.py` - Change from docs to code
- `src/rag/document_index.py` - Rename to chunk_index.py
- `src/agent/graph.py` - Simplify to 3-node flow
- `src/agent/state.py` - Update state schema
- `src/agent/routing.py` - Simplify routing
- `src/agent/nodes/retriever.py` - Adapt for code chunks
- `app.py` - Adapt CLI for code RAG

### 📝 Needs Creation (Critical)
1. **Phase 1** (Indexing):
   - `models/chunk.py` - CodeChunk model
   - `models/retrieval.py` - RetrievalResult model
   - `models/agent.py` - AgentState model
   - `src/indexing/chunker.py` - AST chunking
   - `src/indexing/embedder.py` - Embeddings
   - `src/indexing/index_builder.py` - Build index

2. **Phase 2** (Agent):
   - `src/agent/nodes/router.py` - Query analysis
   - `src/agent/nodes/synthesizer.py` - Answer generation
   - `prompts/router_prompts.py` - Router prompts
   - `prompts/synthesizer_prompts.py` - Synthesizer prompts

3. **Phase 3** (Polish):
   - `src/synthesis/citation_extractor.py` - Citations
   - `src/synthesis/answer_formatter.py` - Formatting
   - `src/retrieval/reranker.py` - Reranking
   - Tests and documentation

## All __init__.py Files Created ✅

Every Python package directory has a proper `__init__.py` file:

```
✅ models/__init__.py
✅ prompts/__init__.py
✅ tests/__init__.py
✅ src/__init__.py
✅ src/indexing/__init__.py
✅ src/retrieval/__init__.py
✅ src/agent/__init__.py
✅ src/agent/nodes/__init__.py
✅ src/synthesis/__init__.py
✅ src/rag/__init__.py
✅ src/llm/__init__.py
✅ src/utils/__init__.py
```

## Next Steps

See `SETUP_STATUS.md` for detailed phase-by-phase implementation plan.

**Ready to start Phase 1: Core Infrastructure** ✅
