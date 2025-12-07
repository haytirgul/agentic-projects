"""Retrieval node with integrated context expansion.

This node performs hybrid search (BM25 + Vector) with weighted RRF and
expands results with contextual information (parent/sibling/imports).

Author: Hay Hoffman
Version: 1.1
"""

import logging
from typing import Any

from settings import (
    CHUNKS_DIR,
    INDEX_DIR,
    TOP_K_PER_REQUEST,
    RRF_BM25_WEIGHT,
    RRF_VECTOR_WEIGHT,
)

from src.agent.state import AgentState
from src.retrieval import ChunkLoader, ContextExpander, HybridRetriever

__all__ = ["retrieval_node"]

logger = logging.getLogger(__name__)

# Module-level singletons for performance
_retriever = None
_expander = None


def _initialize_retrieval_components():
    """Initialize retrieval components (called once at startup)."""
    global _retriever, _expander

    if _retriever is not None:
        logger.debug("Retrieval components already initialized")
        return

    logger.info("Initializing retrieval components...")

    try:
        # Initialize hybrid retriever
        _retriever = HybridRetriever(
            chunks_dir=CHUNKS_DIR,
            index_dir=INDEX_DIR,
            bm25_weight=RRF_BM25_WEIGHT,
            vector_weight=RRF_VECTOR_WEIGHT,
        )

        # Initialize context expander
        chunk_loader = ChunkLoader(chunks_dir=CHUNKS_DIR)
        metadata_index = _retriever.metadata_filter.metadata_index
        _expander = ContextExpander(
            metadata_index=metadata_index,
            chunk_loader=chunk_loader,
        )

        logger.info("[SUCCESS] Retrieval components initialized")

    except Exception as e:
        logger.error(f"Failed to initialize retrieval components: {e}", exc_info=True)
        raise


def retrieval_node(state: AgentState) -> dict[str, Any]:
    """Retrieve and expand chunks with context.

    This node:
    1. Uses RouterOutput to retrieve chunks (hybrid BM25 + Vector)
    2. Applies weighted RRF scoring (BM25: 0.4, Vector: 1.0)
    3. Filters by metadata (source_type, folders, file_patterns)
    4. Expands context (parent class, siblings, imports, child sections)
    5. Returns top-K expanded chunks

    Args:
        state: Current agent state containing router_output

    Returns:
        Updated state with:
            - retrieved_chunks: list of RetrievedChunk dicts
            - expanded_chunks: list of ExpandedChunk dicts
            - retrieval_metadata: Debug info (scores, counts, etc.)

    Raises:
        ValueError: If router_output is missing from state

    Example:
        >>> state = {"router_output": RouterOutput(...)}
        >>> result = retrieval_node(state)
        >>> len(result["expanded_chunks"])
        5
    """
    router_output = state.get("router_output")

    if not router_output:
        raise ValueError("router_output is required for retrieval node")

    logger.info("Starting retrieval with context expansion...")

    # Initialize components (singleton pattern)
    _initialize_retrieval_components()

    # Assert components are initialized (for type checker)
    assert _retriever is not None, "Retriever not initialized"
    assert _expander is not None, "Expander not initialized"

    try:
        # Retrieve chunks using hybrid search
        retrieved_chunks = _retriever.retrieve(
            retrieval_requests=router_output.retrieval_requests,
            top_k=TOP_K_PER_REQUEST,
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks (before expansion)")

        # Expand context for each chunk
        expanded_chunks = []
        for retrieved_chunk in retrieved_chunks:
            expanded = _expander.expand(retrieved_chunk)
            expanded_chunks.append(expanded)

        # Log expansion statistics
        expansion_stats = {
            "with_parent": sum(1 for e in expanded_chunks if e.parent_chunk),
            "with_siblings": sum(1 for e in expanded_chunks if e.sibling_chunks),
            "with_imports": sum(1 for e in expanded_chunks if e.import_chunks),
            "with_children": sum(1 for e in expanded_chunks if e.child_sections),
        }

        logger.info(f"Context expansion: {expansion_stats}")

        # Build metadata
        retrieval_metadata = {
            "num_retrieved": len(retrieved_chunks),
            "num_expanded": len(expanded_chunks),
            "expansion_stats": expansion_stats,
            "top_scores": [chunk.rrf_score for chunk in retrieved_chunks[:5]],
            "source_types": list(
                set(chunk.chunk.source_type for chunk in retrieved_chunks)
            ),
        }

        # Serialize to dicts for state persistence
        retrieved_chunks_dicts = [chunk.model_dump() for chunk in retrieved_chunks]
        expanded_chunks_dicts = [chunk.model_dump() for chunk in expanded_chunks]

        logger.info(
            f"[SUCCESS] Retrieval complete: {len(expanded_chunks)} expanded chunks ready"
        )

        return {
            "retrieved_chunks": retrieved_chunks_dicts,
            "expanded_chunks": expanded_chunks_dicts,
            "retrieval_metadata": retrieval_metadata,
        }

    except Exception as e:
        logger.error(f"Retrieval node failed: {e}", exc_info=True)
        return {
            "error": f"Retrieval failed: {str(e)}",
        }
