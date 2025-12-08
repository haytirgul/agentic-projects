"""Hybrid retrieval combining BM25 and FAISS with weighted RRF.

This module implements the core retrieval architecture featuring:
- Weighted Reciprocal Rank Fusion (0.4/1.0 for BM25/Vector)
- Code-aware BM25 tokenization (camelCase, snake_case splitting)
- Graduated fallback to prevent zero-result failures
- Soft folder filtering (boost, not exclude) to preserve recall
- Metadata-based filtering for source_type and file_patterns
- Parallel multi-query retrieval (max 3 concurrent requests)
- Pre-built BM25 indices per source_type (avoid rebuild per query)
- Pre-compiled regex patterns for tokenization
- Pre-built id->file_path lookup dict
- Async parallel BM25 + FAISS search
- Optimized FAISS vector reconstruction

Author: Hay Hoffman
"""

import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi  

from models.retrieval import RetrievalRequest, RetrievedChunk
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

logger = logging.getLogger(__name__)

__all__ = ["HybridRetriever"]

# Pre-compiled regex patterns for tokenization 
_CAMEL_CASE_PATTERN_1 = re.compile(r'(?<=[a-z])(?=[A-Z])')  # lowercase → uppercase
_CAMEL_CASE_PATTERN_2 = re.compile(r'(?<=[A-Z])(?=[A-Z][a-z])')  # HTTPSServer → HTTPS Server
_UNDERSCORE_PATTERN = re.compile(r'_+')
_NON_ALPHANUM_PATTERN = re.compile(r'[^a-zA-Z0-9x\s]')


class HybridRetriever:
    """Hybrid retrieval combining BM25 and FAISS with weighted RRF.

    Key Features:
    - Weighted RRF (BM25: 0.4, Vector: 1.0) for class chunk recall
    - Code-aware tokenization (camelCase/snake_case splitting)
    - Graduated fallback (4 levels) to prevent zero results
    - SOFT folder filtering: boost scores for folder matches (don't exclude)
    - Pre-built BM25 indices per source_type (avoid rebuild per query)
    - Async parallel search for BM25 + FAISS
    - Parallel multi-query retrieval (max 3 concurrent requests)
    - Pre-built lookup dicts for O(1) access

    Attributes:
        metadata_index: Chunk metadata for filtering
        metadata_filter: MetadataFilter instance (for compatibility)
        faiss_store: FAISS vector store
        chunk_loader: Chunk content loader
        bm25_indices: Pre-built BM25 indices per source_type
        id_to_file_path: Pre-built chunk_id -> file_path mapping
        id_to_tokens: Pre-built chunk_id -> tokenized content mapping
        bm25_weight: Weight for BM25 in RRF scoring
        vector_weight: Weight for vector search in RRF scoring
        folder_boost: Multiplier for chunks matching inferred folders
        _embedding_model: Cached SentenceTransformer model (lazy-loaded)
        _executor: ThreadPoolExecutor for parallel search (BM25 + FAISS)
        _multi_query_executor: ThreadPoolExecutor for parallel multi-query retrieval
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

        # Thread pool for parallel BM25 + FAISS search 
        self._executor = ThreadPoolExecutor(max_workers=2)
        # Thread pool for parallel multi-query retrieval 
        self._multi_query_executor = ThreadPoolExecutor(max_workers=4)

        # Path-based initialization
        if chunks_dir is not None and index_dir is not None:
            from src.retrieval.chunk_loader import ChunkLoader
            from src.retrieval.faiss_store import FAISSStore

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

        # Pre-build optimized lookup structures
        logger.info("Building optimized lookup structures...")
        self.id_to_file_path: dict[str, str] = {}
        self.id_to_tokens: dict[str, list[str]] = {}
        self.bm25_indices: dict[str, BM25Okapi] = {}
        self.bm25_chunk_ids: dict[str, list[str]] = {}  # source_type -> ordered chunk IDs

        self._build_lookup_structures()
        self._build_bm25_indices()

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

    def _build_lookup_structures(self):
        """Build pre-computed lookup dicts for O(1) access.

        Builds:
        - id_to_file_path: chunk_id -> file_path (for folder boost)
        - id_to_tokens: chunk_id -> tokenized content (for BM25)
        """
        for source_type, chunks_meta in self.metadata_index.items():
            for meta in chunks_meta:
                # id -> file_path mapping
                self.id_to_file_path[meta.id] = meta.file_path

                # id -> tokens mapping (pre-tokenize all chunks)
                chunk = self.chunk_loader.load_chunk(meta.id)
                self.id_to_tokens[meta.id] = self._tokenize(chunk.content)

        logger.info(
            f"Built lookup structures: "
            f"{len(self.id_to_file_path)} id->file_path, "
            f"{len(self.id_to_tokens)} id->tokens"
        )

    def _build_bm25_indices(self):
        """Build pre-computed BM25 indices per source_type.

        This avoids rebuilding BM25 index on every search query.
        """
        for source_type, chunks_meta in self.metadata_index.items():
            if not chunks_meta:
                continue

            # Get ordered chunk IDs and their tokens
            chunk_ids = [meta.id for meta in chunks_meta]
            tokenized_corpus = [self.id_to_tokens[cid] for cid in chunk_ids]

            # Build BM25 index
            self.bm25_indices[source_type] = BM25Okapi(tokenized_corpus)
            self.bm25_chunk_ids[source_type] = chunk_ids

            logger.info(f"Built BM25 index for {source_type}: {len(chunk_ids)} chunks")

    def search(
        self,
        request: RetrievalRequest,
        top_k: int = 20
    ) -> list[RetrievedChunk]:
        """Execute hybrid search with soft folder boosting.

        Uses pre-built BM25 indices and parallel search.

        Steps:
        1. Metadata filtering (source_type + file_patterns only)
        2. Parallel: BM25 search + FAISS search
        3. Weighted RRF ranking (0.4/1.0 for BM25/Vector)
        4. Apply folder boost to chunks matching inferred folders
        5. Return top-k chunks

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

        # Convert to set for O(1) lookup (v1.3 optimization)
        candidate_id_set = set(candidate_ids)

        # 2. Parallel BM25 + FAISS search
        bm25_future = self._executor.submit(
            self._bm25_search_optimized,
            request.query,
            candidate_id_set,
            list(request.source_types),  # Convert Literal types to list[str]
            50
        )
        faiss_future = self._executor.submit(
            self._faiss_search_optimized,
            request.query,
            candidate_id_set,
            list(request.source_types),  # Convert Literal types to list[str]
            50
        )

        # Wait for both to complete
        bm25_results = bm25_future.result()
        faiss_results = faiss_future.result()

        # 3. Weighted RRF ranking
        ranked_ids = self._reciprocal_rank_fusion(
            bm25_results=bm25_results,
            faiss_results=faiss_results,
            k=RRF_K,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight
        )

        # 4. Apply folder boost (soft filtering - boost, don't exclude)
        if request.folders:
            ranked_ids = self._apply_folder_boost(
                ranked_ids=ranked_ids,
                folders=request.folders,
                boost_factor=self.folder_boost
            )
            logger.debug(
                f"Applied folder boost ({self.folder_boost}x) for folders: {request.folders}"
            )

        # 5. Get top-k
        top_results = ranked_ids[:top_k]

        # 6. Load full chunks and attach metadata
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

    def _bm25_search_optimized(
        self,
        query: str,
        candidate_id_set: set[str],
        source_types: list[str],
        top_n: int = 50
    ) -> list[tuple[str, float]]:
        """BM25 search using pre-built indices.

        Uses pre-built BM25 indices instead of rebuilding per query.
        Filters results to candidate set after scoring.

        Args:
            query: Search query string
            candidate_id_set: Set of pre-filtered chunk IDs
            source_types: Source types to search
            top_n: Number of results to return

        Returns:
            list of (chunk_id, score) tuples sorted by BM25 score
        """
        tokenized_query = self._tokenize(query)
        all_results: list[tuple[str, float]] = []

        for source_type in source_types:
            if source_type not in self.bm25_indices:
                continue

            bm25 = self.bm25_indices[source_type]
            chunk_ids = self.bm25_chunk_ids[source_type]

            # Get scores for all chunks in this source_type
            scores = bm25.get_scores(tokenized_query)

            # Filter to candidates and collect results
            for idx, (chunk_id, score) in enumerate(zip(chunk_ids, scores)):
                if chunk_id in candidate_id_set:
                    all_results.append((chunk_id, float(score)))

        # Sort by score and return top-N
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_n]

    def _faiss_search_optimized(
        self,
        query: str,
        candidate_id_set: set[str],
        source_types: list[str],
        top_n: int = 50
    ) -> list[tuple[str, float]]:
        """FAISS vector search with optimized reconstruction.

        Uses selective vector reconstruction instead of loading all vectors.

        Args:
            query: Search query string
            candidate_id_set: Set of pre-filtered chunk IDs
            source_types: Source types to search
            top_n: Number of results to return

        Returns:
            list of (chunk_id, similarity_score) tuples
        """
        import faiss

        # Embed query
        query_embedding = self._embed_query(query)

        results: list[tuple[str, float]] = []

        # Search each source type's FAISS index
        for source_type in source_types:
            index = self.faiss_store.load_index(source_type)
            if index is None:
                continue

            # Get ID→position mapping for this source type
            id_to_position = self.faiss_store.get_id_mapping(source_type)

            # Filter candidates to this source type
            source_candidate_ids = [
                cid for cid in candidate_id_set
                if cid in id_to_position
            ]

            if not source_candidate_ids:
                continue

            # Get positions for candidates
            candidate_positions = np.array(
                [id_to_position[cid] for cid in source_candidate_ids],
                dtype=np.int64
            )

            # v1.3: Use reconstruct_batch for selective reconstruction
            # This is more efficient than reconstruct_n(0, ntotal)[positions]
            candidate_vectors = np.zeros(
                (len(candidate_positions), index.d),
                dtype=np.float32
            )
            for i, pos in enumerate(candidate_positions):
                candidate_vectors[i] = index.reconstruct(int(pos))

            # Create temporary index with only candidates
            temp_index = faiss.IndexFlatL2(index.d)
            temp_index.add(candidate_vectors)

            # Search
            distances, indices = temp_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(top_n, len(candidate_positions))
            )

            # Convert back to chunk IDs
            for i, dist in zip(indices[0], distances[0]):
                if i >= 0:  # Valid index
                    chunk_id = source_candidate_ids[i]
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + dist)
                    results.append((chunk_id, float(similarity)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def _tokenize(self, text: str) -> list[str]:
        """Code-aware tokenization for BM25.

        Uses pre-compiled regex patterns for better performance.

        Examples:
        - "HTTPClient" → ["http", "client"]
        - "get_user_by_id" → ["get", "user", "by"]
        - "error 0x884" → ["error", "0x884"]

        Args:
            text: Input text to tokenize

        Returns:
            list of tokens (strings)
        """
        # Step 1: Split camelCase (using pre-compiled patterns)
        if BM25_SPLIT_CAMELCASE:
            text = _CAMEL_CASE_PATTERN_1.sub(' ', text)
            text = _CAMEL_CASE_PATTERN_2.sub(' ', text)

        # Step 2: Split on underscores and non-alphanumeric
        if BM25_SPLIT_SNAKE_CASE:
            text = _UNDERSCORE_PATTERN.sub(' ', text)
        text = _NON_ALPHANUM_PATTERN.sub(' ', text)

        # Step 3: Lowercase and split
        tokens = text.lower().split()

        # Step 4: Filter short tokens (but keep hex codes)
        filtered_tokens = []
        for t in tokens:
            if len(t) > BM25_MIN_TOKEN_LENGTH:
                filtered_tokens.append(t)
            elif BM25_KEEP_HEX_CODES and t.startswith('0x'):
                filtered_tokens.append(t)

        return filtered_tokens

    @classmethod
    def _get_embedding_model(cls):
        """Get cached embedding model (lazy-loaded, singleton pattern).

        Returns:
            SentenceTransformer model instance
        """
        if cls._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            from settings import EMBEDDING_MODEL

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
        """Weighted Reciprocal Rank Fusion with k=60.

        Formula: RRF_score(chunk) = bm25_weight * sum(1/(k + rank_bm25)) +
                                     vector_weight * sum(1/(k + rank_vector))

        Args:
            bm25_results: list of (chunk_id, score) from BM25
            faiss_results: list of (chunk_id, score) from FAISS
            k: RRF constant (default: 60)
            bm25_weight: Weight for BM25 contribution (default: 0.4)
            vector_weight: Weight for Vector contribution (default: 1.0)

        Returns:
            list of (chunk_id, score) tuples sorted by weighted RRF score
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

        return ranked_ids

    def _apply_folder_boost(
        self,
        ranked_ids: list[tuple[str, float]],
        folders: list[str],
        boost_factor: float = 1.3
    ) -> list[tuple[str, float]]:
        """Apply folder boost to chunks matching inferred folders.

        Uses pre-built id_to_file_path dict for O(1) lookup.

        Args:
            ranked_ids: List of (chunk_id, rrf_score) tuples
            folders: List of folder prefixes
            boost_factor: Multiplier for matching chunks (default: 1.3)

        Returns:
            Re-sorted list of (chunk_id, boosted_score) tuples
        """
        boosted_results = []

        for chunk_id, score in ranked_ids:
            # Use pre-built lookup dict
            file_path = self.id_to_file_path.get(chunk_id, "")

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
        """Retrieve chunks from multiple retrieval requests in parallel (v1.4).

        This is the main entry point for the retrieval node, handling
        multiple retrieval requests from the router.

        v1.4: Processes multiple requests in parallel (max 3 concurrent)
        for faster retrieval when router generates multiple queries.

        Args:
            retrieval_requests: List of RetrievalRequest objects from router
            top_k: Number of chunks to return per request

        Returns:
            List of RetrievedChunk objects (deduplicated across requests)
        """
        if not retrieval_requests:
            return []

        # Single request - no parallelism overhead needed
        if len(retrieval_requests) == 1:
            logger.info(f"Single request: {retrieval_requests[0].query[:50]}...")
            return self.search(retrieval_requests[0], top_k=top_k)

        # Multiple requests - process in parallel (max 3 concurrent)
        logger.info(f"Processing {len(retrieval_requests)} requests in parallel...")

        # Submit all search tasks
        futures = [
            self._multi_query_executor.submit(self.search, request, top_k)
            for request in retrieval_requests
        ]

        # Collect results and deduplicate
        all_chunks: dict[str, RetrievedChunk] = {}

        for future, request in zip(futures, retrieval_requests):
            try:
                chunks = future.result()
                logger.debug(f"Request '{request.query[:30]}...' returned {len(chunks)} chunks")

                for chunk in chunks:
                    chunk_id = chunk.chunk.id
                    # Keep higher-scored version if duplicate
                    if chunk_id not in all_chunks or chunk.rrf_score > all_chunks[chunk_id].rrf_score:
                        all_chunks[chunk_id] = chunk

            except Exception as e:
                logger.error(f"Request '{request.query[:30]}...' failed: {e}")
                continue

        # Sort by RRF score and return
        result = sorted(all_chunks.values(), key=lambda x: x.rrf_score, reverse=True)
        logger.info(f"Retrieved {len(result)} unique chunks from {len(retrieval_requests)} parallel requests")

        return result

    def __del__(self):
        """Cleanup thread pools on deletion."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        if hasattr(self, '_multi_query_executor'):
            self._multi_query_executor.shutdown(wait=False)


