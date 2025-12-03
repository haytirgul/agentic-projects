"""
RAG (Retrieval-Augmented Generation) components.

This module provides all retrieval-related functionality including:
- Document indexing and search
- Vector similarity search
- BM25 keyword search
- Web search integration
"""

from .vector_search import VectorSearchEngine, EmbeddingMatch
from .bm25_search import BM25SearchEngine
from .web_search_api import WebSearchEngine, WebSearchResult
from .document_index import DocumentIndex

__all__ = [
    # Search engines
    "VectorSearchEngine",
    "BM25SearchEngine",
    "WebSearchEngine",

    # Data models
    "EmbeddingMatch",
    "WebSearchResult",
    "DocumentIndex",
]
