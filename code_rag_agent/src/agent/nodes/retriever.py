"""
Hybrid retrieval node using VectorDB (ChromaDB) for granular content-block retrieval.

Optimized for low latency with async/await for maximum parallelism.

Uses offline mode only:
- Retrieves from local documentation using VectorDB + BM25 hybrid search
"""

from typing import Any, List, Tuple, Optional
import logging
import asyncio
import pickle
import hashlib
import time
from models.intent_classification import IntentClassification
from src.graph.state import AgentState
from src.rag.vector_search import VectorSearchEngine
from src.rag.hybrid_scorer import HybridScorer
from settings import DATA_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY, AGENT_MODE

logger = logging.getLogger(__name__)

# Initialize at module level (loaded once at startup)
_index = None
_bm25 = None
_vector_search = None
_scorer = None

# Retrieval result cache (in-memory with TTL)
# PERFORMANCE OPTIMIZATION: Cache retrieval results to avoid re-running expensive searches
_retrieval_cache: dict[str, dict[str, Any]] = {}
RETRIEVAL_CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_RETRIEVAL_CACHE_SIZE = 500  # Prevent unbounded growth

# RAG components directory
RAG_COMPONENTS_DIR = DATA_DIR / "rag_components"


def initialize_rag_components():
    """
    Eagerly initialize all RAG components by loading from pickle files.

    This should be called during application startup (e.g., in graph builder)
    to load components in parallel with LLM initialization.

    Raises:
        FileNotFoundError: If RAG component pickle files are not found
        RuntimeError: If loading components fails
    """
    global _index, _bm25, _vector_search, _query_generator, _web_search, _scorer

    if _index is not None:
        logger.info("RAG components already initialized")
        return

    logger.info(f"Initializing RAG components from {RAG_COMPONENTS_DIR}...")

    # Check if RAG components directory exists
    if not RAG_COMPONENTS_DIR.exists():
        error_msg = (
            f"RAG components directory not found: {RAG_COMPONENTS_DIR}\n"
            f"Please run: python scripts/data_pipeline/indexing/build_rag_components.py"
        )
        raise FileNotFoundError(error_msg)

    try:
        # Load DocumentIndex
        index_path = RAG_COMPONENTS_DIR / "document_index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"DocumentIndex not found at {index_path}\n"
                f"Please run: python scripts/data_pipeline/indexing/build_rag_components.py"
            )

        logger.info("Loading DocumentIndex...")
        with open(index_path, 'rb') as f:
            _index = pickle.load(f)
        all_docs = _index.get_all_documents()
        logger.info(f"✓ DocumentIndex loaded ({len(all_docs)} documents)")

    except Exception as e:
        raise RuntimeError(f"Failed to load DocumentIndex: {e}") from e

    try:
        # Load BM25 Index
        bm25_path = RAG_COMPONENTS_DIR / "bm25_index.pkl"
        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}\n"
                f"Please run: python scripts/data_pipeline/indexing/build_rag_components.py"
            )

        logger.info("Loading BM25 index...")
        with open(bm25_path, 'rb') as f:
            _bm25 = pickle.load(f)
        logger.info("✓ BM25 index loaded")

    except Exception as e:
        raise RuntimeError(f"Failed to load BM25 index: {e}") from e

    # Initialize VectorDB
    vector_db_path = DATA_DIR / "vector_db"

    try:
        logger.info("Initializing VectorDB search (FAISS)...")
        _vector_search = VectorSearchEngine(
            persist_directory=vector_db_path,
            collection_name="documentation_embeddings",
            embedding_model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY if EMBEDDING_MODEL.startswith("models/") else None,
            content_preview_chars=300,
            enable_block_level=True,
            document_index=_index,
        )

        stats = _vector_search.get_collection_stats()
        if stats['total_embeddings'] == 0:
            _vector_search = None
        else:
            logger.info(f"✓ VectorDB loaded: {stats['total_embeddings']} embeddings")
            logger.info(f"  Content-block-level: {stats['granularity_breakdown'].get('content_block', 0)}")

    except Exception as e:
        _vector_search = None


    # Initialize hybrid scorer
    _scorer = HybridScorer()
    logger.info("✓ Hybrid scorer initialized")

    logger.info(f"✓ All RAG components initialized successfully (mode: {AGENT_MODE})")


# ============================================================================
# Async Search Functions for Maximum Parallelism
# ============================================================================

async def _run_vector_search_async(vector_query: str, intent: IntentClassification) -> List[Any]:
    """
    Run VectorDB search asynchronously using clean natural language query.

    Args:
        vector_query: Clean natural language query (no keyword pollution)
        intent: Intent classification for metadata filtering
    """
    if _vector_search is None:
        logger.debug("Skipping VectorDB search (not initialized)")
        return []

    # TIMING: Track vector search
    start_time = time.time()
    try:
        # Run in thread pool since VectorDB operations are blocking
        loop = asyncio.get_event_loop()

        # Filter out "general" from framework list (means no filtering)
        framework_filter = None
        if intent.framework:
            if isinstance(intent.framework, list):
                filtered = [f for f in intent.framework if f != "general"]
                framework_filter = filtered if filtered else None
            elif intent.framework != "general":
                framework_filter = intent.framework

        results = await loop.run_in_executor(
            None,
            lambda: _vector_search.search(
                query=vector_query,
                top_k=30,
                framework=framework_filter,
                language=intent.language,
                topic=None,
                granularity_boost=True,
            )
        )

        # TIMING: Log vector search time
        elapsed = time.time() - start_time
        from src.utils.timing_logger import get_timing_logger
        get_timing_logger().log_timing(
            "retrieval.vector_search",
            elapsed,
            {"num_results": len(results), "framework": framework_filter}
        )

        logger.debug(f"VectorDB returned {len(results)} results")
        return results
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Vector search failed after {elapsed:.3f}s: {e}")
        return []


async def _run_bm25_search_async(bm25_query: str) -> List[Tuple[Any, float]]:
    """
    Run BM25 keyword search asynchronously using keyword-enriched query.

    Args:
        bm25_query: Query with extracted keywords and topics (optimized for keyword matching)
    """
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: _bm25.search(bm25_query, top_k=50)
        )
        logger.debug(f"BM25 returned {len(results)} results")
        return results
    except Exception as e:
        return []



async def _run_all_searches_async(
    vector_query: str,
    bm25_query: str,
    intent: IntentClassification
) -> Tuple[List, List]:
    """
    Run vector and BM25 searches in parallel for maximum performance.

    Args:
        vector_query: Clean natural language query for semantic search
        bm25_query: Keyword-enriched query for BM25 lexical search
        intent: Intent classification for metadata filtering

    Returns:
        Tuple of (vector_results, bm25_results)
    """
    # Execute both searches concurrently with optimized queries
    results = await asyncio.gather(
        _run_vector_search_async(vector_query, intent),
        _run_bm25_search_async(bm25_query),
        return_exceptions=True  # Don't fail if one search fails
    )

    # Handle any exceptions
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append([])
        else:
            final_results.append(result)

    return tuple(final_results)


# ============================================================================
# Deduplication Functions
# ============================================================================

def _deduplicate_by_text(results: List[Any]) -> List[Any]:
    """
    Remove exact duplicate text across all results.

    This catches cases where identical content appears in different documents
    (e.g., duplicated docs across /oss/ and /python/ directories).

    Args:
        results: List of EmbeddingMatch objects or dicts with 'matched_text'

    Returns:
        Deduplicated list, keeping highest-scoring matches

    Example:
        >>> results = [match1, match2, match3]  # match2 and match3 have same text
        >>> deduped = _deduplicate_by_text(results)
        >>> len(deduped)
        2  # match3 removed as duplicate
    """
    seen_texts = set()
    deduplicated = []
    duplicates_removed = 0

    for result in results:
        # Get text content
        if hasattr(result, 'matched_text'):
            text = result.matched_text
            doc_path = getattr(result, 'document_path', 'unknown')
        elif isinstance(result, dict):
            text = result.get('matched_text', '')
            doc_path = result.get('document_path', 'unknown')
        else:
            # Unknown format, keep it
            deduplicated.append(result)
            continue

        # Normalize: lowercase, collapse whitespace, strip
        normalized = ' '.join(text.lower().split())

        # Skip if seen before (exact match after normalization)
        if normalized in seen_texts:
            logger.debug(f"Skipping duplicate text from {doc_path}")
            duplicates_removed += 1
            continue

        seen_texts.add(normalized)
        deduplicated.append(result)

    if duplicates_removed > 0:
        logger.info(f"Text deduplication: {len(results)} → {len(deduplicated)} results ({duplicates_removed} duplicates removed)")

    return deduplicated


# ============================================================================
# Retrieval Cache Functions
# ============================================================================

def _get_cache_key(cleaned_request: str, intent: IntentClassification) -> str:
    """Generate cache key for retrieval results."""
    # Include query, framework, and topics in cache key
    framework_str = str(intent.framework) if intent.framework else "none"
    topics_str = ":".join(sorted(intent.topics)) if intent.topics else "none"

    cache_input = f"{cleaned_request}:{framework_str}:{topics_str}"
    return hashlib.md5(cache_input.encode()).hexdigest()


def _get_cached_retrieval(cleaned_request: str, intent: IntentClassification) -> Optional[List[dict]]:
    """Check if retrieval results are cached for this query."""
    cache_key = _get_cache_key(cleaned_request, intent)

    if cache_key in _retrieval_cache:
        cached = _retrieval_cache[cache_key]
        age = time.time() - cached['timestamp']

        if age < RETRIEVAL_CACHE_TTL_SECONDS:
            logger.info(f"Retrieval cache HIT (age: {age:.1f}s) for query: {cleaned_request[:50]}...")
            return cached['documents']
        else:
            # Expired, remove it
            del _retrieval_cache[cache_key]
            logger.debug(f"Retrieval cache entry expired for query: {cleaned_request[:50]}...")

    return None


def _set_retrieval_cache(cleaned_request: str, intent: IntentClassification, documents: List[dict]):
    """Store retrieval results in cache with TTL."""
    cache_key = _get_cache_key(cleaned_request, intent)

    _retrieval_cache[cache_key] = {
        'documents': documents,
        'timestamp': time.time()
    }

    # Evict oldest entries if cache is too large
    if len(_retrieval_cache) > MAX_RETRIEVAL_CACHE_SIZE:
        oldest_key = min(_retrieval_cache.items(), key=lambda x: x[1]['timestamp'])[0]
        del _retrieval_cache[oldest_key]
        logger.debug(f"Evicted oldest retrieval cache entry (size: {len(_retrieval_cache)})")


# ============================================================================
# Main Hybrid Retrieval Node
# ============================================================================

def hybrid_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    Hybrid retrieval using VectorDB + BM25 + Web (async optimized).

    PERFORMANCE OPTIMIZED:
    - All searches run in parallel with asyncio.gather()
    - Vector, BM25, and Web searches execute concurrently
    - Minimizes total latency by maximizing parallelism

    Process:
    1. Extract intent and build search query
    2. Run ALL 3 searches in parallel (vector, BM25, web)
    3. Combine scores with weighted formula
    4. Filter by framework/language
    5. Return top-5 results

    Returns:
        Dict with retrieved_documents and retrieval_metadata
    """
    # Ensure components initialized
    if _index is None:
        error_msg = (
            "RAG components not initialized! "
            "This should have been done during graph building. "
            "Please check initialization errors."
        )
        raise RuntimeError(error_msg)

    # Extract from state
    intent = state.get("intent_result")
    cleaned_request = state.get("cleaned_request") or state.get("user_input", "")

    if not intent:
        intent = IntentClassification(
            intent_type="factual_lookup",
            framework=None,
            language=None,
            keywords=[],
            topics=[],
            requires_rag=True
        )

    logger.info(f"Starting hybrid retrieval for: {cleaned_request[:100]}")

    # PERFORMANCE OPTIMIZATION: Check cache first
    cached_docs = _get_cached_retrieval(cleaned_request, intent)
    if cached_docs:
        return {
            "retrieved_documents": cached_docs,
            "retrieval_metadata": {
                "cache_hit": True,
                "vector_query": cleaned_request,
                "bm25_query": "",
                "mode": AGENT_MODE,
                "vector_results": 0,
                "bm25_results": 0,
                "web_results": 0,  # Always 0 in offline-only mode
                "top_scores": [d["relevance_score"] for d in cached_docs],
                "granularity_breakdown": {},
                "source_breakdown": {},
            }
        }

    # Build optimized queries for each search method
    # Vector search: Use clean natural language (best for semantic similarity)
    vector_query = cleaned_request

    # BM25 search: Add keywords and topics (best for keyword matching)
    bm25_query_parts = []
    if intent.keywords:
        bm25_query_parts.extend(intent.keywords[:5])
    if intent.topics:
        bm25_query_parts.extend(intent.topics)
    bm25_query = " ".join(bm25_query_parts)

    logger.debug(f"Vector query (semantic): {vector_query[:100]}")
    logger.debug(f"BM25 query (keywords): {bm25_query[:100]}")

    # Run ALL searches in parallel using asyncio
    logger.debug(f"Running searches in parallel (mode: {AGENT_MODE})...")

    # Create event loop if not exists, or use existing one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Execute searches concurrently with optimized queries
    vector_results, bm25_results = loop.run_until_complete(
        _run_all_searches_async(vector_query, bm25_query, intent)
    )

    # Deduplicate vector results by text content (CRITICAL FIX for duplicate results)
    vector_results = _deduplicate_by_text(vector_results)


    # Use HybridScorer to combine and rank results
    # RRF is faster (no normalization) and often more accurate than weighted combination
    retrieved_docs = _scorer.score_and_rank_rrf(
        vector_results=vector_results,
        bm25_results=bm25_results,
        web_results=[],  # No web results in offline-only mode
        intent=intent,
        top_k=5
    )

    # Log stats
    granularity_counts = {}
    source_counts = {'offline': 0, 'web': 0}
    for doc in retrieved_docs:
        granularity_counts[doc.granularity] = granularity_counts.get(doc.granularity, 0) + 1
        source = doc.retrieval_metadata.get('source', 'offline')
        source_counts[source] = source_counts.get(source, 0) + 1

    avg_score = sum(d.relevance_score for d in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
    logger.info(f"Retrieved {len(retrieved_docs)} results (avg score: {avg_score:.2f})")
    logger.info(f"  Sources: offline={source_counts['offline']}, web={source_counts.get('web', 0)}")
    logger.info(f"  Granularity: {granularity_counts}")

    # Convert RetrievedDocument objects to dicts for msgpack serialization
    retrieved_docs_dicts = [doc.model_dump() for doc in retrieved_docs]

    # PERFORMANCE OPTIMIZATION: Cache results for future identical queries
    _set_retrieval_cache(cleaned_request, intent, retrieved_docs_dicts)

    return {
        "retrieved_documents": retrieved_docs_dicts,
        "retrieval_metadata": {
            "cache_hit": False,
            "vector_query": vector_query,
            "bm25_query": bm25_query,
            "mode": AGENT_MODE,
            "vector_results": len(vector_results),
            "bm25_results": len(bm25_results),
            "web_results": 0,  # Always 0 in offline-only mode
            "top_scores": [d.relevance_score for d in retrieved_docs],
            "granularity_breakdown": granularity_counts,
            "source_breakdown": source_counts,
        }
    }
