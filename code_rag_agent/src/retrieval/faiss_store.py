"""FAISS vector store for semantic search.

This module provides lazy-loaded FAISS indices for code, markdown, and text
chunks. Indices are stored on disk and loaded on-demand to minimize RAM usage.

Author: Hay Hoffman
Version: 1.1
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from settings import FAISS_EMBEDDING_DIM

logger = logging.getLogger(__name__)

__all__ = ["FAISSStore"]


class FAISSStore:
    """FAISS vector store with separate indices per source_type.

    Design:
    - FAISS indices stored on disk (reduces RAM usage)
    - Lazy loading: Load index only when needed for search
    - Separate index per source_type for efficient filtering
    - ID-to-position mapping for chunk retrieval

    Attributes:
        index_dir: Directory containing FAISS index files
        indices: Loaded FAISS indices (lazy-loaded)
        id_mappings: Chunk ID to FAISS position mappings
    """

    def __init__(self, index_dir: Path):
        """Initialize FAISS store.

        Args:
            index_dir: Directory containing FAISS index files
                      (e.g., data/index/)
        """
        self.index_dir = Path(index_dir)

        # Index file paths
        self.index_paths = {
            "code": self.index_dir / "code_faiss.index",
            "markdown": self.index_dir / "markdown_faiss.index",
            "text": self.index_dir / "text_faiss.index"
        }

        # ID mapping file paths
        self.mapping_paths = {
            "code": self.index_dir / "code_id_mapping.json",
            "markdown": self.index_dir / "markdown_id_mapping.json",
            "text": self.index_dir / "text_id_mapping.json"
        }

        # Loaded indices (lazy-loaded)
        self.indices: dict[str, faiss.Index | None] = {
            "code": None,
            "markdown": None,
            "text": None
        }

        # ID mappings (chunk_id -> FAISS position)
        self.id_mappings: dict[str, dict[str, int]] = {
            "code": {},
            "markdown": {},
            "text": {}
        }

        # Position to ID reverse mapping (FAISS position -> chunk_id)
        self.position_to_id: dict[str, dict[int, str]] = {
            "code": {},
            "markdown": {},
            "text": {}
        }

        # Check which indices exist
        self._check_available_indices()

    def _check_available_indices(self):
        """Check which FAISS indices are available on disk."""
        for source_type, path in self.index_paths.items():
            if path.exists():
                logger.info(f"Found FAISS index for {source_type}: {path}")
            else:
                logger.warning(f"FAISS index not found for {source_type}: {path}")

    def load_index(self, source_type: str) -> faiss.Index | None:
        """Load FAISS index for a source type (lazy loading).

        Also loads the corresponding ID mapping file if available.

        Args:
            source_type: One of "code", "markdown", "text"

        Returns:
            Loaded FAISS index or None if not found

        Raises:
            ValueError: If source_type is invalid
        """
        if source_type not in ["code", "markdown", "text"]:
            raise ValueError(
                f"Invalid source_type: {source_type}. "
                f"Must be one of: code, markdown, text"
            )

        # Return cached index if already loaded
        if self.indices[source_type] is not None:
            return self.indices[source_type]

        # Load from disk
        index_path = self.index_paths[source_type]
        if not index_path.exists():
            logger.warning(f"FAISS index not found: {index_path}")
            return None

        try:
            logger.info(f"Loading FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))
            self.indices[source_type] = index
            logger.info(
                f"Loaded FAISS index for {source_type}: "
                f"{index.ntotal} vectors, dim={index.d}"
            )

            # Load ID mapping if available
            self._load_id_mapping(source_type)

            return index

        except Exception as e:
            logger.error(f"Failed to load FAISS index from {index_path}: {e}")
            return None

    def _load_id_mapping(self, source_type: str) -> None:
        """Load ID mapping from JSON file.

        Args:
            source_type: One of "code", "markdown", "text"
        """
        mapping_path = self.mapping_paths[source_type]
        if not mapping_path.exists():
            logger.warning(f"ID mapping not found: {mapping_path}")
            return

        try:
            with open(mapping_path, encoding="utf-8") as f:
                # JSON keys are strings, convert position keys to int
                raw_mapping = json.load(f)
                for pos_str, chunk_id in raw_mapping.items():
                    pos = int(pos_str)
                    self.position_to_id[source_type][pos] = chunk_id
                    self.id_mappings[source_type][chunk_id] = pos

            logger.info(
                f"Loaded ID mapping for {source_type}: "
                f"{len(self.position_to_id[source_type])} entries"
            )

        except Exception as e:
            logger.error(f"Failed to load ID mapping from {mapping_path}: {e}")

    def get_index(self, source_type: str) -> faiss.Index | None:
        """Get FAISS index for a source type (loads if needed).

        Args:
            source_type: One of "code", "markdown", "text"

        Returns:
            FAISS index or None if not available
        """
        return self.load_index(source_type)

    def set_id_mapping(
        self,
        source_type: str,
        chunk_id: str,
        position: int
    ):
        """Set mapping between chunk ID and FAISS position.

        Args:
            source_type: One of "code", "markdown", "text"
            chunk_id: Unique chunk identifier
            position: Position in FAISS index
        """
        self.id_mappings[source_type][chunk_id] = position
        self.position_to_id[source_type][position] = chunk_id

    def get_id_mapping(self, source_type: str) -> dict[str, int]:
        """Get chunk ID to FAISS position mapping.

        Args:
            source_type: One of "code", "markdown", "text"

        Returns:
            Dict mapping chunk_id -> FAISS position
        """
        return self.id_mappings[source_type]

    def get_chunk_id_by_position(
        self,
        source_type: str,
        position: int
    ) -> str | None:
        """Get chunk ID by FAISS position.

        Args:
            source_type: One of "code", "markdown", "text"
            position: Position in FAISS index

        Returns:
            Chunk ID or None if not found
        """
        return self.position_to_id[source_type].get(position)

    def search(
        self,
        source_type: str,
        query_vector: np.ndarray,
        top_k: int = 50
    ) -> list[tuple[str, float]]:
        """Search FAISS index for similar vectors.

        Args:
            source_type: One of "code", "markdown", "text"
            query_vector: Query embedding vector (384-dim)
            top_k: Number of results to return

        Returns:
            list of (chunk_id, similarity_score) tuples

        Raises:
            ValueError: If index not loaded or query vector has wrong shape
        """
        index = self.get_index(source_type)
        if index is None:
            logger.warning(f"No FAISS index for {source_type}")
            return []

        # Validate query vector shape
        if query_vector.shape != (FAISS_EMBEDDING_DIM,):
            raise ValueError(
                f"Query vector must be {FAISS_EMBEDDING_DIM}-dimensional, "
                f"got {query_vector.shape}"
            )

        # Reshape for FAISS (expects 2D array)
        query_vector = query_vector.reshape(1, -1).astype('float32')

        # Search
        distances, indices = index.search(query_vector, min(top_k, index.ntotal))

        # Convert to (chunk_id, similarity) pairs
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + dist)

            # Get chunk ID from position
            chunk_id = self.get_chunk_id_by_position(source_type, int(idx))
            if chunk_id:
                results.append((chunk_id, float(similarity)))

        return results

    def get_stats(self) -> dict:
        """Get statistics about FAISS indices.

        Returns:
            Dict with index statistics
        """
        stats = {}
        for source_type in ["code", "markdown", "text"]:
            index = self.indices.get(source_type)
            if index:
                stats[source_type] = {
                    "loaded": True,
                    "total_vectors": index.ntotal,
                    "dimension": index.d,
                    "id_mappings": len(self.id_mappings[source_type])
                }
            else:
                stats[source_type] = {
                    "loaded": False,
                    "index_exists": self.index_paths[source_type].exists()
                }

        return stats
