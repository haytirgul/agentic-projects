"""Retrieval module for the Code RAG Agent.

This module implements the complete retrieval pipeline as specified in
RETRIEVAL_ARCHITECTURE.md v1.1, including:

- Hybrid search (BM25 + FAISS with weighted RRF)
- Fast path routing for simple queries
- Metadata filtering with graduated fallback
- Code-aware BM25 tokenization
- FAISS vector search
- Chunk loading and caching
- Context expansion with parent/sibling/import enrichment

Author: Hay Hoffman
"""

from src.retrieval.chunk_loader import ChunkLoader
from src.retrieval.context_expander import ContextExpander
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.fast_path_router import FastPathRouter
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.metadata_filter import ChunkMetadata, MetadataFilter

__all__ = [
    "HybridRetriever",
    "FastPathRouter",
    "MetadataFilter",
    "ChunkMetadata",
    "FAISSStore",
    "ChunkLoader",
    "ContextExpander",
]
