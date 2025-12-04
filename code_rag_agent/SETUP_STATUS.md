# Code RAG Agent - Setup Status

## ✅ Files Copied from RAG_agent

### Infrastructure (Adapted)
- [x] `settings.py` - Configuration adapted for code RAG (removed security/web search, added code-specific settings)
- [x] `requirements.txt` - Dependencies adapted (removed security ML models, added AST parsing)
- [x] `.gitignore` - Git ignore patterns for code RAG project
- [x] `.env.example` - Environment variable template

### Models
- [x] `models/conversation.py` - Conversation memory models (simplified from RAG_agent)
- [x] `models/__init__.py` - Package initialization

### RAG Components (Needs Adaptation)
- [x] `src/rag/vector_search.py` - FAISS vector search (**needs adaptation for code chunks**)
- [x] `src/rag/bm25_search.py` - BM25 keyword search (**needs adaptation for code**)
- [x] `src/rag/hybrid_scorer.py` - Hybrid scoring with RRF reranking
- [x] `src/rag/document_index.py` - In-memory document index (**needs renaming to chunk_index**)

### LLM Components
- [x] `src/llm/llm.py` - LLM factory with Gemini support
- [x] `src/llm/workers.py` - Structured output parsing workers

### LangGraph Infrastructure (Needs Major Adaptation)
- [x] `src/agent/graph.py` - LangGraph builder (**needs simplification to 3 nodes**)
- [x] `src/agent/state.py` - Agent state TypedDict (**needs adaptation for code chunks**)
- [x] `src/agent/routing.py` - Conditional routing logic (**needs simplification**)

### Graph Nodes (Needs Adaptation/Creation)
- [x] `src/agent/nodes/conversation_memory.py` - Memory tracking node
- [x] `src/agent/nodes/retriever.py` - Hybrid retrieval node (**needs adaptation**)
- [ ] `src/agent/nodes/router.py` - **NEEDS TO BE CREATED**
- [ ] `src/agent/nodes/synthesizer.py` - **NEEDS TO BE CREATED**

### Utilities
- [x] `src/utils/logger.py` - Logging utilities

### CLI
- [x] `app.py` - Interactive CLI (**needs adaptation for code RAG workflow**)

---

## 🚧 Files That Need To Be Created

### Code Chunking (Critical - Phase 1)
- [ ] `src/indexing/chunker.py` - AST-based code chunking
- [ ] `src/indexing/embedder.py` - Embedding generation for code chunks
- [ ] `src/indexing/index_builder.py` - FAISS/ChromaDB index builder

### Code-Specific Models (Critical - Phase 1)
- [ ] `models/chunk.py` - CodeChunk Pydantic model
- [ ] `models/retrieval.py` - RetrievalResult Pydantic model
- [ ] `models/agent.py` - AgentState Pydantic model

### Retrieval
- [ ] `src/retrieval/reranker.py` - Reranking with evaluation metrics

### Synthesis
- [ ] `src/synthesis/citation_extractor.py` - Extract file:line citations
- [ ] `src/synthesis/answer_formatter.py` - Format answers with citations

### Prompts (Critical - Phase 2)
- [ ] `prompts/router_prompts.py` - Router node prompts (8-section structure)
- [ ] `prompts/retriever_prompts.py` - Retriever node prompts
- [ ] `prompts/synthesizer_prompts.py` - Synthesizer node prompts

### Graph Nodes (Critical - Phase 2)
- [ ] `src/agent/nodes/router.py` - Query analysis and routing
- [ ] `src/agent/nodes/synthesizer.py` - Answer generation with citations

### Testing
- [ ] `tests/test_chunker.py` - Chunking tests
- [ ] `tests/test_retrieval.py` - Retrieval tests
- [ ] `tests/test_agent.py` - End-to-end agent tests

### Documentation
- [ ] `README.md` - Setup and usage instructions
- [ ] `ARCHITECTURE.md` - Design decisions and rationale

---

## 🔧 Major Adaptations Needed

### 1. Vector Search (`src/rag/vector_search.py`)
**Current**: Searches documentation with 3-level granularity (document/section/content_block)
**Needed**: Search code chunks with metadata (file_path, line_range, function/class context)

**Changes Required**:
- Replace `EmbeddingMatch` dataclass with code-specific fields
- Update metadata filters for framework → file_path, language → always Python
- Simplify granularity from 3 levels to 1 level (code chunks)

### 2. BM25 Search (`src/rag/bm25_search.py`)
**Current**: Keyword search on documentation
**Needed**: Keyword search on code with special handling for:
- Function/class names
- Import statements
- Docstrings
- Code identifiers

### 3. LangGraph State (`src/agent/state.py`)
**Current**: Documentation RAG state with intent classification
**Needed**: Code RAG state with:
- `user_query: str`
- `current_chunks: List[CodeChunk]`
- `retrieval_history: List[RetrievalResult]`
- `conversation_memory: ConversationMemory`
- `answer: str`
- `citations: List[str]`

### 4. Graph Builder (`src/agent/graph.py`)
**Current**: Multi-node graph (security → intent → retrieval → agent → output)
**Needed**: Simplified 3-node graph:
```
router → retriever → synthesizer (with conversation memory)
```

### 5. CLI (`app.py`)
**Current**: Documentation Q&A chatbot
**Needed**: Code analysis chatbot with:
- httpx repository context
- Code-specific prompts
- File:line citation display

---

## 📋 Next Steps (Priority Order)

### Phase 1: Core Infrastructure (Days 1-2)
1. **Create Code Models**
   - `models/chunk.py` - CodeChunk with file_path, line_range, content, metadata
   - `models/retrieval.py` - RetrievalResult with hybrid scores
   - `models/agent.py` - AgentState for LangGraph

2. **Implement Code Chunking**
   - `src/indexing/chunker.py` - AST-based chunking with Python `ast` module
   - `src/indexing/embedder.py` - Generate embeddings for code chunks
   - `src/indexing/index_builder.py` - Build ChromaDB/FAISS index

3. **Adapt RAG Components**
   - Modify `src/rag/vector_search.py` for code chunks
   - Modify `src/rag/bm25_search.py` for code search
   - Update `src/rag/document_index.py` → `src/rag/chunk_index.py`

### Phase 2: Agent Flow (Days 3-4)
1. **Create Graph Nodes**
   - `src/agent/nodes/router.py` - Parse query, plan retrieval
   - `src/agent/nodes/synthesizer.py` - Generate answer with citations

2. **Create Prompts**
   - `prompts/router_prompts.py` - Query analysis prompts
   - `prompts/synthesizer_prompts.py` - Answer generation prompts

3. **Adapt LangGraph**
   - Simplify `src/agent/graph.py` to 3-node flow
   - Update `src/agent/state.py` with code-specific state
   - Simplify `src/agent/routing.py` for basic routing

### Phase 3: Synthesis & Citations (Day 5)
1. **Citation System**
   - `src/synthesis/citation_extractor.py` - Extract file:line references
   - `src/synthesis/answer_formatter.py` - Format with citations

2. **Testing**
   - `tests/test_chunker.py` - Verify chunking logic
   - `tests/test_retrieval.py` - Test hybrid retrieval
   - `tests/test_agent.py` - End-to-end tests

### Phase 4: Polish (Day 6)
1. **CLI & Documentation**
   - Adapt `app.py` for code RAG workflow
   - Write `README.md` with setup instructions
   - Write `ARCHITECTURE.md` with design rationale

2. **Example Outputs**
   - Generate answers for 3 sample queries
   - Document retrieval logs

---

## ⚠️ Known Issues to Address

1. **Import Paths**: Many copied files have imports like:
   - `from models.intent_classification import ...` (not needed for code RAG)
   - `from settings import SECURITY_ENABLED` (removed from settings)

   **Action**: Will need to clean up imports during adaptation

2. **Documentation-Specific Logic**: Files contain logic for:
   - Framework filtering (LangChain vs LangGraph)
   - Section-level embeddings
   - Topic classification

   **Action**: Remove or replace with code-specific logic (file paths, line ranges)

3. **ChromaDB vs FAISS**:
   - RAG_agent uses ChromaDB
   - PRD mentions FAISS preference

   **Decision**: Start with ChromaDB (already copied), can add FAISS later if needed

---

## 🎯 Success Criteria

✅ **Phase 1 Complete When**:
- httpx repo can be chunked into code blocks
- Chunks have proper metadata (file, lines, context)
- ChromaDB index is built successfully
- Basic retrieval works (query → relevant chunks)

✅ **Phase 2 Complete When**:
- 3-node LangGraph flow executes
- Router parses queries correctly
- Retriever returns relevant code chunks
- Synthesizer generates answers with citations

✅ **Phase 3 Complete When**:
- All citations are in `file.py:start-end` format
- Citations point to correct code locations
- Agent refuses when evidence is weak

✅ **Phase 4 Complete When**:
- `python app.py` starts interactive CLI
- Sample queries return correct answers
- README/ARCHITECTURE docs are complete
- Project runs with one-command setup

---

**Last Updated**: 2025-12-04
**Status**: Infrastructure copied, ready for adaptation
