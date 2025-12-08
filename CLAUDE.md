# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a multi-project workspace for agentic Python applications. Each project follows a consistent architecture based on LLM-powered agents with structured patterns for prompting, data validation, and workflow orchestration.

**Active Projects:**
- `code_rag_agent/`: Code analysis RAG system (modern pyproject.toml setup)
- `melingo-test-task/`: Order processing agent using LangChain
- `opsfleet-task/`: Documentation RAG system using LangGraph
- `RAG_agent/`: General RAG system implementation
- `rules/`: Shared coding standards and architectural guidelines (8 comprehensive guides)
- `scripts/`: Shared utility scripts

## Core Architecture Principles

### Directory Structure Pattern

All agent projects follow this modular structure:

```
project_root/
├── src/              # Core application logic (pipeline, tools, LLM clients)
├── models/           # Pydantic data models (domain entities, requests/responses)
├── prompts/          # Prompt templates as Python code (NOT strings in logic)
├── data/             # Static data and assets (input/output directories)
├── documents/        # Project documentation (architecture, decisions)
├── tests/            # Test suite (mirrors src/ structure)
├── settings.py       # Configuration management (uses dotenv and pathlib)
├── .env              # Environment variables (NOT committed)
└── requirements.txt  # Python dependencies
```

### Key Design Patterns

**1. Prompt-as-Code**
- All prompts live in `prompts/` as Python modules
- Use functions to generate dynamic prompts with context injection
- Follow 8-section structure: Persona → Task → Requirements → Tools → Examples → Pitfalls → Output Schema → Validation
- Separate examples into `prompts/examples.py` for cleaner templates

**2. Structured Outputs (MANDATORY for JSON)**
- **ALWAYS** use `.with_structured_output()` for JSON responses from LLMs
- Use Pydantic `BaseModel` for ALL data schemas
- Never manually parse JSON from LLM responses
- Benefits: automatic validation, type safety, built-in retry/repair loops

**3. Fuzzy-First Pattern**
- Don't use LLM for everything
- Use fuzzy matching (`rapidfuzz`) for known entities before calling LLM
- Fallback chain: Fuzzy matching → LLM → Error handling

**4. LangChain/LangGraph Integration**
- Use `langchain_core.messages` for typed message objects (`SystemMessage`, `HumanMessage`)
- For complex agents: LangGraph with state management and conditional routing
- For simple flows: Direct LangChain chains
- **ALWAYS stream** user-facing responses: use `llm.stream()` or `llm.astream()`

## Development Commands

### Testing

**Run all tests:**
```bash
# From project root (e.g., melingo-test-task/ or opsfleet-task/)
pytest tests/ -v
```

**Run specific test:**
```bash
pytest tests/test_graph.py::test_specific_function -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

**Run single test file (melingo pattern):**
```bash
python test_single.py
```

### Code Quality

**Modern approach (Ruff replaces Black + isort):**
```bash
# Format code (replaces Black)
ruff format .

# Sort imports and fix linting issues
ruff check . --fix

# Type checking
mypy src/ --strict --ignore-missing-imports
```

**Legacy approach (if project uses Black):**
```bash
black . --line-length=100
isort . --profile=black --line-length=100
ruff check . --fix
```

### Running Agents

**Melingo (Order Processing):**
```bash
python agent.py
```

**Opsfleet (RAG System):**
```bash
# Build RAG index (first time)
python data_processing/build_rag_index.py

# Run graph agent
python src/graph/builder.py
```

## Code Style Requirements

**Python Version:** 3.10+ (prefer 3.11+ for performance)

**Mandatory Patterns:**
- Strict type hints on all function arguments and return values
- Use modern Python 3.10+ type syntax: `X | Y` instead of `Union[X, Y]`, `X | None` instead of `Optional[X]`
- Use built-in generics: `list[str]` instead of `list[str]`
- Google-style docstrings for all public functions/classes
- `__all__` exports defined in each module
- Logger initialized per module: `logger = logging.getLogger(__name__)`
- All external API calls wrapped in `try-except` blocks
- Pathlib for all file operations (never use string paths)

**Type Hints Best Practices:**
- Use domain models when available: `def process(user: User) -> Result:`
- Use primitives when appropriate: `def parse(data: str) -> dict[str, Any]:`
- Use `Protocol` for structural typing and interfaces
- Use `collections.abc` types (`Sequence`, `Mapping`) for flexibility

**Naming Conventions:**
- Directories: `snake_case` (e.g., `data_processing`)
- Files: `snake_case` (e.g., `order_handler.py`)
- Classes: `PascalCase` (e.g., `OrderParser`)
- Functions/Variables: `snake_case` (e.g., `process_order`)
- Constants: `UPPER_CASE` (e.g., `MAX_RETRIES`)

**Import Order:**
1. Standard library
2. Third-party packages (pydantic, langchain, etc.)
3. Local application modules (src, models, prompts)

## Project-Specific Details

### Melingo Test Task

**Purpose:** Grocery order processing agent with natural language parsing

**Key Components:**
- `agent.py`: Main entry point with `process_order()` function
- `src/pipeline.py`: Orchestration (parse → match → validate → calculate)
- `src/parser.py`: Structured output parser for LLM responses
- `src/tools.py`: Agent tools (e.g., `search_items_tool` for catalog search)
- `models/order.py`: Order domain models
- `prompts/order_parsing.py`: Main order parsing prompt
- `prompts/item_matching.py`: Item catalog matching prompt

**Dependencies:**
- LangChain for LLM orchestration
- OpenAI and Anthropic model support
- RapidFuzz for fuzzy string matching

### Code RAG Agent

**Purpose:** Code analysis and understanding RAG system (MODERN SETUP)

**Key Characteristics:**
- Uses `pyproject.toml` for all configuration (modern Python packaging)
- Ruff for both formatting and linting (line-length: 88)
- Python 3.11+ target
- Clean separation: `settings.py` (runtime config) and `const.py` (constants)

**Architecture:**
- Hybrid retrieval for code chunks
- Semantic code search with embeddings
- Context-aware code explanations

### Opsfleet Task

**Purpose:** Documentation RAG system with LangGraph agent workflow

**Key Components:**
- `src/graph/builder.py`: LangGraph workflow definition
- `src/graph/edges.py`: Conditional routing logic
- `src/nodes/`: Individual graph nodes (gateway, intent classification)
- `data_processing/build_rag_index.py`: Vector database ingestion
- `data_processing/langchain_parser.py`: Documentation parser for LangChain docs
- `data_processing/langgraph_parser.py`: Documentation parser for LangGraph docs
- `models/gateway.py`: Security gateway validation models
- `models/intent.py`: Intent classification and clarification models

**Architecture:**
- Multi-stage LangGraph: Gateway validation → Intent classification → RAG processing
- ChromaDB for vector storage
- Hybrid search: BM25 + vector embeddings with reranking
- Human-in-the-loop via graph interrupts for clarifications

**Dependencies:**
- LangGraph for state machine workflows
- ChromaDB for vector database
- Google Gemini for embeddings
- rank_bm25 for keyword search

## RAG Implementation Notes

When working on RAG systems in this repository:

1. **Always use hybrid search:** Combine dense (vector) + sparse (BM25) retrieval
2. **Implement reranking:** Retrieve top-25 to top-50, rerank to final top-5
3. **Query rewriting:** Never pass raw user input to retriever directly
4. **Evaluation is mandatory:** Track faithfulness, context precision, answer relevancy
5. **Citation instructions:** LLM must cite sources explicitly
6. **"I don't know" policy:** LLM must refuse to answer if context insufficient

## Environment Configuration

All projects use `settings.py` for centralized configuration:

**Pattern:**
```python
from os import getenv
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# API Keys (fail fast if missing)
OPENAI_API_KEY = getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing")
```

**Required `.env` variables:**
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Claude models (melingo)
- `GOOGLE_API_KEY`: For Gemini models (opsfleet)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Git Workflow

**Commit Message Format (Conventional Commits):**
```
<type>(<scope>): <subject>

<body>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Examples:**
```bash
feat(rag): add hybrid search with BM25 and vector embeddings
fix(parser): handle search metadata blocks in llms.txt
docs: add setup instructions and architecture overview
```

**Before Committing:**
1. Format code: `ruff format .` (or `black . && isort .` for legacy projects)
2. Lint and fix: `ruff check . --fix`
3. Run tests: `pytest tests/`
4. Type check: `mypy src/ --ignore-missing-imports`
5. Check for secrets: `git diff --cached | grep -i "api_key"`

## VS Code Configuration

Projects should include `.vscode/settings.json` and `.vscode/launch.json`:

**Key settings:**
- Auto-format on save with Ruff (modern) or Black (legacy)
- Auto-organize imports on save with Ruff
- Auto-fix linting issues (Ruff)
- Line length: 100 characters (88 for code_rag_agent)
- Pytest integration enabled
- Python interpreter: `.venv/bin/python`

**Debug configurations:**
- Python: Current File
- Python: Run Tests
- Python: Run Agent/Main Script

## Important Rules Reference

Detailed guidelines are in the `rules/` directory (recently updated with modern best practices):

- [01-structure.md](rules/01-structure.md:1): Project structure, naming conventions, pyproject.toml vs requirements.txt
- [02-style.md](rules/02-style.md:1): Code style with modern type hints, Ruff as unified formatter/linter
- [03-agentic.md](rules/03-agentic.md:1): LLM agent patterns, structured outputs, circuit breakers
- [04-config.md](rules/04-config.md:1): Environment and configuration management with validation
- [05-prompting.md](rules/05-prompting.md:1): 8-section prompt structure, streaming requirements
- [06-rag.md](rules/06-rag.md:1): RAG standards with hybrid search, evaluation metrics
- [07-git.md](rules/07-git.md:1): Version control workflow, conventional commits
- [08-vscode.md](rules/08-vscode.md:1): VS Code workspace configuration with Ruff

**Key Updates (2025-12-06):**
- Ruff now replaces Black + isort in all new projects
- Modern Python 3.10+ type syntax (`X | Y`, `list[str]`)
- Structured output with `.with_structured_output()` is MANDATORY for JSON
- Streaming responses is MANDATORY for user-facing output
- Complete validation utilities and helper functions

When in doubt about implementation details, consult the relevant rule file first.
