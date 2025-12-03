# Documentation Index

**Project**: Opsfleet Documentation Assistant
**Last Updated**: 2025-11-30

---

## Document Organization

This directory contains all core project documentation, organized by purpose and audience.

---

## For Users

### Getting Started

**[../README.md](../README.md)** - Start here!
- Project overview and key features
- Quick start guide (installation, setup, first run)
- Usage examples and common queries
- Configuration reference
- Troubleshooting guide

**[task.md](task.md)** - Original Assignment
- Project requirements and deliverables
- Technical specifications
- Example questions the agent should handle
- Evaluation criteria

---

## For Developers

### Development Guide

**[DEV_GUIDE.md](DEV_GUIDE.md)** - Comprehensive Developer Guide
- **Start here for development**: Complete guide for developers and AI agents
- Project structure explained in detail
- Core components with code examples
- Data flow through the system
- Key design patterns (Prompt-as-Code, Structured Outputs, Hybrid Search)
- Development workflow (adding nodes, prompts, intent types)
- Testing strategy and guidelines
- Extension points (adding retrievers, new features)
- Troubleshooting and debugging
- Performance guidelines and optimization tips

**Target Audience**: Developers, AI agents (like Claude), contributors

---

### Architecture Documentation

**[ARCHITECTURE.md](ARCHITECTURE.md)** - System Architecture (Version 2.0)
- Executive summary
- High-level system architecture
- Graph structure and node inventory
- Complete data flow with examples (clear question, vague question, invalid request)
- Node specifications (preprocessing, intent classification, etc.)
- State management and persistence
- Implementation history and optimizations
- Next steps (RAG implementation, multi-turn conversations, streaming)
- File structure reference
- Configuration and troubleshooting

**Status**: Describes the system BEFORE RAG implementation
**Target Audience**: Architects, senior developers

---

**[ARCHITECTURE_AND_LEARNING_GUIDE.md](ARCHITECTURE_AND_LEARNING_GUIDE.md)** - Extended Learning Guide
- Similar to ARCHITECTURE.md but with more educational content
- Learning objectives and patterns
- Detailed explanations for each concept
- Best practices and anti-patterns

**Target Audience**: Developers learning LangGraph and agentic patterns

---

### RAG Implementation

**[RAG_IMPLEMENTATION_PRD.md](RAG_IMPLEMENTATION_PRD.md)** - RAG Product Requirements (Version 2.0)
- **Most detailed RAG documentation**: 66KB comprehensive guide
- Executive summary and design rationale
- Complete system architecture for hybrid RAG
- Implementation tasks with code examples:
  - In-memory document index
  - BM25 search engine
  - Embedding-based semantic search
  - Fuzzy keyword matching
  - Hybrid retrieval node
  - Web search integration (online mode)
- Data structures and models
- Testing strategy and evaluation datasets
- Success metrics and rollout plan
- Performance optimizations (parallel search execution)
- Appendices with detailed comparisons and tuning parameters

**Status**: Production implementation guide
**Target Audience**: Developers implementing or extending RAG functionality

---

**[RAG.md](RAG.md)** - RAG Concepts and Overview
- RAG fundamentals and best practices
- Retrieval strategies
- Evaluation approaches
- Shorter conceptual overview compared to PRD

**Target Audience**: Developers learning RAG concepts

---

### Graph Structure

**[GRAPH_DIAGRAM.md](GRAPH_DIAGRAM.md)** - Visual Graph Structure
- LangGraph node and edge diagrams
- Visual representation of the state machine
- Flow charts for different scenarios

**Target Audience**: Visual learners, architects

---

**[nodes.md](nodes.md)** - Node Specifications
- Detailed specification for each graph node
- Input/output contracts
- Implementation notes

**Target Audience**: Developers implementing or debugging nodes

---

### Design Decisions

**[react-vs-current-architecture.md](react-vs-current-architecture.md)** - Architecture Comparison
- ReAct pattern vs current multi-stage approach
- Pros/cons of each approach
- Why the current architecture was chosen

**Target Audience**: Architects, decision makers

---

**[agent_loop_pattern.md](agent_loop_pattern.md)** - Agent Loop Patterns
- Agent tool execution patterns
- Human-in-the-loop implementation
- Clarification workflows

**Target Audience**: Developers working on agent tools

---

## Documentation Quick Reference

### By Use Case

| I want to... | Read this |
|--------------|-----------|
| Get started using the agent | [README.md](../README.md) |
| Understand the requirements | [task.md](task.md) |
| Start developing | [DEV_GUIDE.md](DEV_GUIDE.md) |
| Understand the architecture | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Implement RAG features | [RAG_IMPLEMENTATION_PRD.md](RAG_IMPLEMENTATION_PRD.md) |
| Visualize the graph | [GRAPH_DIAGRAM.md](GRAPH_DIAGRAM.md) |
| Learn RAG concepts | [RAG.md](RAG.md) |
| Implement a node | [nodes.md](nodes.md) |
| Understand design choices | [react-vs-current-architecture.md](react-vs-current-architecture.md) |
| Implement agent tools | [agent_loop_pattern.md](agent_loop_pattern.md) |

---

### By Audience

| Audience | Recommended Reading Order |
|----------|---------------------------|
| **End Users** | 1. README.md |
| **New Developers** | 1. README.md → 2. DEV_GUIDE.md → 3. ARCHITECTURE.md |
| **AI Agents** | 1. DEV_GUIDE.md → 2. RAG_IMPLEMENTATION_PRD.md → 3. ARCHITECTURE.md |
| **Architects** | 1. ARCHITECTURE.md → 2. react-vs-current-architecture.md → 3. RAG_IMPLEMENTATION_PRD.md |
| **RAG Engineers** | 1. RAG_IMPLEMENTATION_PRD.md → 2. RAG.md → 3. DEV_GUIDE.md |

---

## Document Status

| Document | Status | Last Updated | Size |
|----------|--------|--------------|------|
| README.md | ✅ Production | 2025-11-30 | 14KB |
| DEV_GUIDE.md | ✅ Production | 2025-11-30 | 23KB |
| ARCHITECTURE.md | ⚠️ Needs Update | 2025-11-25 | 30KB |
| RAG_IMPLEMENTATION_PRD.md | ✅ Production | 2025-11-28 | 66KB |
| ARCHITECTURE_AND_LEARNING_GUIDE.md | ✅ Complete | 2025-11-27 | 48KB |
| GRAPH_DIAGRAM.md | ✅ Complete | 2025-11-25 | 16KB |
| RAG.md | ✅ Complete | 2025-11-27 | 38KB |
| nodes.md | ✅ Complete | 2025-11-28 | 9KB |
| agent_loop_pattern.md | ✅ Complete | 2025-11-28 | 14KB |
| react-vs-current-architecture.md | ✅ Complete | 2025-11-28 | 21KB |
| task.md | ✅ Reference | 2025-11-20 | 5KB |

**Legend:**
- ✅ Production: Current and accurate
- ⚠️ Needs Update: Contains outdated information (pre-RAG implementation)
- ✅ Complete: Comprehensive and current
- ✅ Reference: Historical reference, not updated

---

## Removed Documentation

The following intermediate testing and debug documentation has been removed to keep the docs folder clean:

### Removed Files (2025-11-30)

- `BUGFIX_FRAMEWORK_FILTERING.md` - Bug fix record
- `BUGFIX_HORIZONTAL_RULE_TRUNCATION.md` - Bug fix record
- `BUGFIX_MISSING_CONTENT.md` - Bug fix record
- `CHATBOT_MEMORY_FIX.md` - Temporary fix documentation
- `CHATBOT_README.md` - Outdated chatbot readme
- `CLEANUP_SUMMARY.md` - Cleanup record
- `CONVERSATION_CONTEXT_ROUTING.md` - Implementation details (superseded by ARCHITECTURE.md)
- `CONVERSATION_MEMORY.md` - Implementation details (superseded by ARCHITECTURE.md)
- `DEBUGGING_SETUP.md` - Debug helper
- `debugging-guide.md` - Debug helper
- `EMBEDDING_CACHE_GUIDE.md` - Implementation details (superseded by RAG_IMPLEMENTATION_PRD.md)
- `EMBEDDING_IMPROVEMENTS_SUMMARY.md` - Implementation log
- `ENHANCEMENT_CONTENT_EXPANSION.md` - Implementation log
- `HUGGING_FACE_MIGRATION.md` - Intermediate migration guide
- `IMPLEMENTATION_SUMMARY.md` - Implementation log
- `markdown_to_json_conversion.md` - Data processing details
- `QUICKSTART_VECTOR_SEARCH.md` - Intermediate guide
- `VECTOR_SEARCH_GUIDE.md` - Intermediate guide (superseded by RAG_IMPLEMENTATION_PRD.md)
- `VERSION_3_UPDATES.md` - Version update log

**Total Removed**: 19 files

These files documented intermediate development steps, bug fixes, and temporary implementations that are no longer relevant. The core knowledge from these files has been consolidated into the production documentation above.

---

## Contributing to Documentation

### Adding New Documentation

1. Create the document in the `documents/` directory
2. Add entry to this INDEX.md
3. Link from relevant sections in README.md or DEV_GUIDE.md
4. Update the "Document Status" table

### Updating Existing Documentation

1. Update the document
2. Update "Last Updated" date
3. If major changes, update version number
4. Review cross-references in other docs

### Documentation Standards

- **File naming**: Use UPPER_CASE.md for major docs, lower_case.md for supporting docs
- **Headers**: Use ATX-style headers (# ## ###)
- **Code blocks**: Always specify language (```python, ```bash)
- **Links**: Use relative links within the project
- **TOC**: Include table of contents for docs > 5KB
- **Metadata**: Include version, date, status, audience at the top

---

## External Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Google Gemini API](https://ai.google.dev/docs)

---

**Questions?** Start with [DEV_GUIDE.md](DEV_GUIDE.md) or [README.md](../README.md)
