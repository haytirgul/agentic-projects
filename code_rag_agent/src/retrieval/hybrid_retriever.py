"""Hybrid retrieval combining BM25 and FAISS with weighted RRF.

This module implements the core retrieval architecture specified in
RETRIEVAL_ARCHITECTURE.md v1.2, featuring:
- Weighted Reciprocal Rank Fusion (0.4/1.0 for BM25/Vector)
- Code-aware BM25 tokenization (camelCase, snake_case splitting)
- Graduated fallback to prevent zero-result failures
- Soft folder filtering (boost, not exclude) to preserve recall
- Metadata-based filtering for source_type and file_patterns

Author: Hay Hoffman
Version: 1.2
"""

import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from settings import (
    BM25_KEEP_HEX_CODES,
    BM25_MIN_TOKEN_LENGTH,
    BM25_SPLIT_CAMELCASE,
    BM25_SPLIT_SNAKE_CASE,
    RRF_BM25_WEIGHT,
    RRF_FOLDER_BOOST,
    RRF_K,
    RRF_VECTOR_WEIGHT,
)
from models.retrieval import RetrievalRequest, RetrievedChunk
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """Hybrid retrieval combining BM25 and FAISS with weighted RRF (v1.2).

    Key Features (v1.2):
    - Weighted RRF (BM25: 0.4, Vector: 1.0) for class chunk recall
    - Code-aware tokenization (camelCase/snake_case splitting)
    - Graduated fallback (4 levels) to prevent zero results
    - SOFT folder filtering: boost scores for folder matches (don't exclude)
    - Metadata filtering for source_type and file_patterns only

    v1.2 Change: Folder filtering is now SOFT (boost, not exclude).
    This prevents LLM folder inference from reducing recall when it guesses wrong.

    Attributes:
        metadata_index: Chunk metadata for filtering
        metadata_filter: MetadataFilter instance (for compatibility)
        faiss_store: FAISS vector store
        chunk_loader: Chunk content loader
        bm25_cache: Cached BM25 indices for reuse
        bm25_weight: Weight for BM25 in RRF scoring
        vector_weight: Weight for vector search in RRF scoring
        folder_boost: Multiplier for chunks matching inferred folders
        _embedding_model: Cached SentenceTransformer model (lazy-loaded)
    """

    # Class-level cached embedding model (shared across instances)
    _embedding_model = None

    def __init__(
        self,
        chunks_dir: Path | None = None,
        index_dir: Path | None = None,
        bm25_weight: float = RRF_BM25_WEIGHT,
        vector_weight: float = RRF_VECTOR_WEIGHT,
        folder_boost: float = RRF_FOLDER_BOOST,
        *,
        # Alternative: pass pre-built components directly
        metadata_index: dict[str, list] | None = None,
        faiss_store=None,
        chunk_loader=None,
    ):
        """Initialize hybrid retriever.

        Supports two initialization modes:
        1. Path-based: Pass chunks_dir and index_dir to build components internally
        2. Component-based: Pass pre-built metadata_index, faiss_store, chunk_loader

        Args:
            chunks_dir: Directory containing chunk JSON files (path-based init)
            index_dir: Directory containing FAISS indices (path-based init)
            bm25_weight: Weight for BM25 in RRF scoring (default: 0.4)
            vector_weight: Weight for vector search in RRF scoring (default: 1.0)
            folder_boost: Multiplier for chunks matching inferred folders (default: 1.3)
            metadata_index: Pre-built mapping of source_type -> list of chunk metadata
            faiss_store: Pre-built FAISSStore instance
            chunk_loader: Pre-built ChunkLoader instance
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.folder_boost = folder_boost

        # Path-based initialization
        if chunks_dir is not None and index_dir is not None:
            from src.retrieval.chunk_loader import ChunkLoader
            from src.retrieval.faiss_store import FAISSStore
            from src.retrieval.metadata_filter import ChunkMetadata

            logger.info(f"Initializing HybridRetriever from paths: {chunks_dir}, {index_dir}")

            # Build chunk loader
            self.chunk_loader = ChunkLoader(chunks_dir=chunks_dir)

            # Build FAISS store
            self.faiss_store = FAISSStore(index_dir=index_dir)

            # Build metadata index from loaded chunks
            self.metadata_index = self._build_metadata_index(self.chunk_loader)

        # Component-based initialization
        elif metadata_index is not None and faiss_store is not None and chunk_loader is not None:
            logger.info("Initializing HybridRetriever from pre-built components")
            self.metadata_index = metadata_index
            self.faiss_store = faiss_store
            self.chunk_loader = chunk_loader

        else:
            raise ValueError(
                "HybridRetriever requires either (chunks_dir, index_dir) or "
                "(metadata_index, faiss_store, chunk_loader)"
            )

        # Create a MetadataFilter-compatible wrapper for backward compatibility
        self.metadata_filter = _MetadataFilterCompat(self.metadata_index)

        # BM25 indices (built on-demand from filtered corpus)
        self.bm25_cache: dict[str, BM25Okapi] = {}

        logger.info(
            f"HybridRetriever initialized: "
            f"{sum(len(v) for v in self.metadata_index.values())} chunks, "
            f"bm25_weight={bm25_weight}, vector_weight={vector_weight}, "
            f"folder_boost={folder_boost}"
        )

    def _build_metadata_index(self, chunk_loader) -> dict[str, list]:
        """Build metadata index from chunk loader.

        Args:
            chunk_loader: ChunkLoader with loaded chunks

        Returns:
            Dict mapping source_type -> list of ChunkMetadata
        """
        from src.retrieval.metadata_filter import ChunkMetadata

        metadata_index: dict[str, list] = {}

        for source_type, chunks in chunk_loader.chunks_by_source.items():
            metadata_index[source_type] = []

            for chunk in chunks:
                # Extract metadata fields from chunk
                meta = ChunkMetadata(
                    id=chunk.id,
                    source_type=chunk.source_type,
                    filename=chunk.filename,
                    file_path=chunk.file_path,
                    chunk_type=getattr(chunk, 'chunk_type', None),
                    name=getattr(chunk, 'name', None),
                    parent_context=getattr(chunk, 'parent_context', None),
                    start_line=getattr(chunk, 'start_line', None),
                    end_line=getattr(chunk, 'end_line', None),
                    heading_level=getattr(chunk, 'heading_level', None),
                )
                metadata_index[source_type].append(meta)

        logger.info(
            f"Built metadata index: "
            f"{', '.join(f'{k}={len(v)}' for k, v in metadata_index.items())}"
        )

        return metadata_index

    def search(
        self,
        request: RetrievalRequest,
        top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Execute hybrid search with soft folder boosting (v1.2).

        v1.2 Update: Folder filtering is now SOFT (boost, not exclude).
        This prevents LLM folder inference from reducing recall.

        Steps:
        1. Metadata filtering (source_type + file_patterns only)
        2. BM25 search (on filtered corpus with code-aware tokenization)
        3. FAISS search (on filtered vectors)
        4. Weighted RRF ranking (0.4/1.0 for BM25/Vector)
        5. Apply folder boost to chunks matching inferred folders
        6. Return top-k chunks

        Fallback chain (for file_patterns only):
        - Attempt 1: With file_patterns
        - Attempt 2: Drop file_patterns if zero results
        - Attempt 3: Emergency (search all source_types)

        Args:
            request: Retrieval request with query and filters
            top_k: Number of chunks to return (default: 20)

        Returns:
            list of RetrievedChunk objects sorted by relevance
        """
        from src.retrieval.metadata_filter import MetadataFilter

        # 1. Metadata filtering (source_type + file_patterns, NOT folders)
        candidate_ids = MetadataFilter(
            self.metadata_index,
            request
        ).apply()

        # Fallback 1: Drop file_patterns if zero results
        if not candidate_ids and request.file_patterns:
            logger.info(
                f"File patterns {request.file_patterns} yielded 0 results. "
                f"Relaxing to source_type-only filtering."
            )
            relaxed_request = request.model_copy(update={"file_patterns": []})
            candidate_ids = MetadataFilter(
                self.metadata_index,
                relaxed_request
            ).apply()

        # Fallback 2: Emergency - search all source_types
        if not candidate_ids:
            logger.warning(
                "All filters yielded 0 results. Emergency fallback: searching all source_types."
            )
            emergency_request = RetrievalRequest(
                query=request.query,
                source_types=["code", "markdown", "text"],
                folders=[],
                file_patterns=[],
                reasoning=request.reasoning + " [EMERGENCY FALLBACK]"
            )
            candidate_ids = MetadataFilter(
                self.metadata_index,
                emergency_request
            ).apply()

        # If still no candidates, return empty
        if not candidate_ids:
            logger.error(f"No candidates found even after full relaxation for query: {request.query}")
            return []

        logger.info(
            f"Filtered to {len(candidate_ids)} candidates "
            f"for query: {request.query}"
        )

        # Build chunk_id -> file_path mapping for folder boost
        chunk_file_paths = self._get_chunk_file_paths(candidate_ids)

        # 2. BM25 search (code-aware tokenization)
        bm25_results = self._bm25_search(
            query=request.query,
            candidate_ids=candidate_ids,
            top_n=50
        )

        # 3. FAISS search
        faiss_results = self._faiss_search(
            query=request.query,
            candidate_ids=candidate_ids,
            source_types=request.source_types,
            top_n=50
        )

        # 4. Weighted RRF ranking (use instance weights)
        ranked_ids = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            k=RRF_K,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight
        )

        # 5. Apply folder boost (soft filtering - boost, don't exclude)
        if request.folders:
            ranked_ids = self._apply_folder_boost(
                ranked_ids=ranked_ids,
                folders=request.folders,
                chunk_file_paths=chunk_file_paths,
                boost_factor=self.folder_boost
            )
            logger.info(
                f"Applied folder boost ({self.folder_boost}x) for folders: {request.folders}"
            )

        # 6. Get top-k
        top_results = ranked_ids[:top_k]

        # 7. Load full chunks and attach metadata
        retrieved_chunks = []
        for chunk_id, rrf_score in top_results:
            chunk = self.chunk_loader.load_chunk(chunk_id)
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk,
                    rrf_score=rrf_score,
                    request_reasoning=request.reasoning,
                    source_query=request.query
                )
            )

        return retrieved_chunks

    def _bm25_search(
        self,
        query: str,
        candidate_ids: list[str],
        top_n: int = 50
    ) -> list[tuple[str, float]]:
        """BM25 search on filtered corpus with code-aware tokenization (v1.1).

        Strategy: Build temporary BM25 index from filtered chunks
        (fast for small candidate sets, no need to maintain full index)

        Args:
            query: Search query string
            candidate_ids: Pre-filtered chunk IDs to search
            top_n: Number of results to return

        Returns:
            list of (chunk_id, score) tuples sorted by BM25 score
        """
        # Load content for candidate chunks
        candidate_chunks = []
        for chunk_id in candidate_ids:
            chunk = self.chunk_loader.load_chunk(chunk_id)
            candidate_chunks.append({
                "id": chunk.id,
                "content": chunk.content
            })

        # Tokenize corpus (v1.1: code-aware tokenization)
        tokenized_corpus = [
            self._tokenize(chunk["content"])
            for chunk in candidate_chunks
        ]

        # Build BM25 index
        bm25 = BM25Okapi(tokenized_corpus)

        # Search
        tokenized_query = self._tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        # Get top-N results
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = [
            (candidate_chunks[i]["id"], scores[i])
            for i in top_indices
        ]

        return results

    def _tokenize(self, text: str) -> list[str]:
        """Code-aware tokenization for BM25 (v1.1).

        v1.1 Update: Handles camelCase, snake_case, and preserves hex codes.

        Examples:
        - "HTTPClient" → ["http", "client"]
        - "get_user_by_id" → ["get", "user", "by"]
        - "error 0x884" → ["error", "0x884"]

        Strategy:
        1. Split camelCase boundaries (lowercase→uppercase)
        2. Replace underscores and punctuation with spaces
        3. Lowercase and filter short tokens (except hex codes)

        Args:
            text: Input text to tokenize

        Returns:
            list of tokens (strings)
        """
        # Step 1: Split camelCase (e.g., 'HTTPClient' → 'HTTP Client')
        if BM25_SPLIT_CAMELCASE:
            text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # lowercase → uppercase
            text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)  # HTTPSServer → HTTPS Server

        # Step 2: Split on underscores and non-alphanumeric (but keep 0x prefix)
        if BM25_SPLIT_SNAKE_CASE:
            text = re.sub(r'_+', ' ', text)  # Underscores → spaces
        text = re.sub(r'[^a-zA-Z0-9x\s]', ' ', text)  # Keep 'x' for hex

        # Step 3: Lowercase and split
        tokens = text.lower().split()

        # Step 4: Filter short tokens (but keep hex codes like '0x')
        filtered_tokens = []
        for t in tokens:
            if len(t) > BM25_MIN_TOKEN_LENGTH:  # Standard filter
                filtered_tokens.append(t)
            elif BM25_KEEP_HEX_CODES and t.startswith('0x'):  # Exception for hex
                filtered_tokens.append(t)

        return filtered_tokens

    def _faiss_search(
        self,
        query: str,
        candidate_ids: list[str],
        source_types: list[str] | list,
        top_n: int = 50
    ) -> list[tuple[str, float]]:
        """FAISS vector search on filtered candidates.

        Steps:
        1. Load relevant FAISS indices (by source_type)
        2. Get FAISS positions for candidate IDs
        3. Build temporary index from candidate vectors
        4. Search and convert to chunk IDs

        Args:
            query: Search query string
            candidate_ids: Pre-filtered chunk IDs to search
            source_types: Source types to search (code, markdown, text)
            top_n: Number of results to return

        Returns:
            list of (chunk_id, similarity_score) tuples
        """
        # Embed query
        query_embedding = self._embed_query(query)

        results = []

        # Search each source type's FAISS index
        for source_type in source_types:
            index = self.faiss_store.get_index(source_type)
            if index is None:
                continue

            # Get ID→position mapping for this source type
            id_to_position = self.faiss_store.get_id_mapping(source_type)

            # Filter candidates to this source type
            source_candidate_ids = [
                cid for cid in candidate_ids
                if cid in id_to_position
            ]

            if not source_candidate_ids:
                continue

            # Get positions and vectors for candidates
            candidate_positions = [id_to_position[cid] for cid in source_candidate_ids]
            candidate_vectors = index.reconstruct_n(0, index.ntotal)[candidate_positions]

            # Create candidate ID mapping
            candidate_id_map = {pos: cid for pos, cid in zip(candidate_positions, source_candidate_ids)}

            # Create temporary index with only candidates
            import faiss
            temp_index = faiss.IndexFlatL2(index.d)
            temp_index.add(candidate_vectors)

            # Search
            distances, indices = temp_index.search(
                query_embedding.reshape(1, -1),
                min(top_n, len(candidate_positions))
            )

            # Convert back to chunk IDs
            for i, dist in zip(indices[0], distances[0]):
                original_position = candidate_positions[i]
                chunk_id = candidate_id_map[original_position]
                # Convert L2 distance to similarity score
                similarity = 1 / (1 + dist)
                results.append((chunk_id, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    @classmethod
    def _get_embedding_model(cls):
        """Get cached embedding model (lazy-loaded, singleton pattern).

        Returns:
            SentenceTransformer model instance
        """
        if cls._embedding_model is None:
            from settings import EMBEDDING_MODEL
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            cls._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")

        return cls._embedding_model

    @classmethod
    def preload_embedding_model(cls) -> None:
        """Pre-load the embedding model during initialization.

        Call this during init_project.py to download and cache the model
        so queries don't trigger downloads at runtime.
        """
        cls._get_embedding_model()

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding (uses cached model).

        Args:
            query: Query string to embed

        Returns:
            Numpy array of embedding vector
        """
        model = self._get_embedding_model()
        embedding = model.encode(query)
        return np.asarray(embedding)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[tuple[str, float]],
        faiss_results: list[tuple[str, float]],
        k: int = 60,
        bm25_weight: float = 0.4,
        vector_weight: float = 1.0
    ) -> list[tuple[str, float]]:
        """Weighted Reciprocal Rank Fusion with k=60 (v1.1).

        v1.1 Update: Weighted RRF favoring semantic search to improve recall
        for short class signature chunks (which BM25 under-ranks due to length bias).

        Formula: RRF_score(chunk) = bm25_weight * sum(1/(k + rank_bm25)) +
                                     vector_weight * sum(1/(k + rank_vector))

        Weighting Rationale:
        - BM25 (0.4): Good for exact terms, but biased by document length
        - Vector (1.0): Better for intent, especially for short signature chunks

        Args:
            bm25_results: list of (chunk_id, score) from BM25
            faiss_results: list of (chunk_id, score) from FAISS
            k: RRF constant (default: 60)
            bm25_weight: Weight for BM25 contribution (default: 0.4)
            vector_weight: Weight for Vector contribution (default: 1.0)

        Returns:
            list of (chunk_id, score) tuples sorted by weighted RRF score (highest first)
        """
        rrf_scores: dict[str, float] = defaultdict(float)

        # Weighted BM25 contribution
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] += bm25_weight * (1.0 / (k + rank))

        # Weighted Vector contribution
        for rank, (chunk_id, _) in enumerate(faiss_results, start=1):
            rrf_scores[chunk_id] += vector_weight * (1.0 / (k + rank))

        # Sort by weighted RRF score
        ranked_ids = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return list of (chunk_id, score) tuples for score access
        return ranked_ids

    def _get_chunk_file_paths(
        self,
        candidate_ids: list[str]
    ) -> dict[str, str]:
        """Get file_path for each chunk ID from metadata index.

        Args:
            candidate_ids: List of chunk IDs

        Returns:
            Dict mapping chunk_id -> file_path
        """
        chunk_file_paths = {}

        # Build a quick lookup from all metadata
        for source_type, chunks_meta in self.metadata_index.items():
            for meta in chunks_meta:
                if meta.id in candidate_ids:
                    chunk_file_paths[meta.id] = meta.file_path

        return chunk_file_paths

    def _apply_folder_boost(
        self,
        ranked_ids: list[tuple[str, float]],
        folders: list[str],
        chunk_file_paths: dict[str, str],
        boost_factor: float = 1.3
    ) -> list[tuple[str, float]]:
        """Apply folder boost to chunks matching inferred folders (v1.2).

        Chunks whose file_path starts with any of the specified folders
        get their RRF score multiplied by boost_factor.

        This is SOFT filtering - non-matching chunks are NOT excluded,
        just ranked lower.

        Args:
            ranked_ids: List of (chunk_id, rrf_score) tuples
            folders: List of folder prefixes (e.g., ["httpx/_transports/"])
            chunk_file_paths: Mapping of chunk_id -> file_path
            boost_factor: Multiplier for matching chunks (default: 1.3)

        Returns:
            Re-sorted list of (chunk_id, boosted_score) tuples

        Example:
            >>> ranked = [("chunk1", 0.5), ("chunk2", 0.4)]
            >>> paths = {"chunk1": "httpx/_client.py", "chunk2": "httpx/_transports/base.py"}
            >>> folders = ["httpx/_transports/"]
            >>> self._apply_folder_boost(ranked, folders, paths, 1.3)
            [("chunk2", 0.52), ("chunk1", 0.5)]  # chunk2 boosted and reranked
        """
        boosted_results = []

        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")

            # Check if file_path matches any folder
            matches_folder = any(
                file_path.startswith(folder)
                for folder in folders
            )

            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score

            boosted_results.append((chunk_id, boosted_score))

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x[1], reverse=True)

        return boosted_results

    def retrieve(
        self,
        retrieval_requests: list[RetrievalRequest],
        top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from multiple retrieval requests.

        This is the main entry point for the retrieval node, handling
        multiple retrieval requests from the router.

        Args:
            retrieval_requests: List of RetrievalRequest objects from router
            top_k: Number of chunks to return per request

        Returns:
            List of RetrievedChunk objects (deduplicated across requests)
        """
        all_chunks: dict[str, RetrievedChunk] = {}  # Dedupe by chunk_id

        for request in retrieval_requests:
            logger.info(f"Processing retrieval request: {request.query[:50]}...")

            chunks = self.search(request, top_k=top_k)

            for chunk in chunks:
                chunk_id = chunk.chunk.id
                # Keep higher-scored version if duplicate
                if chunk_id not in all_chunks or chunk.rrf_score > all_chunks[chunk_id].rrf_score:
                    all_chunks[chunk_id] = chunk

        # Sort by RRF score and return
        result = sorted(all_chunks.values(), key=lambda x: x.rrf_score, reverse=True)
        logger.info(f"Retrieved {len(result)} unique chunks from {len(retrieval_requests)} requests")

        return result


class _MetadataFilterCompat:
    """Compatibility wrapper providing metadata_index attribute.

    This allows retrieval.py to access metadata_index via retriever.metadata_filter.metadata_index
    for backward compatibility with ContextExpander.
    """

    def __init__(self, metadata_index: dict[str, list]):
        self.metadata_index = metadata_index
