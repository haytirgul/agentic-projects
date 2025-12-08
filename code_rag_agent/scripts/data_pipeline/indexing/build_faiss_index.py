"""Build FAISS indices from processed chunks.

This script:
1. Loads all chunks from data/processed/all_chunks.json
2. Groups them by source_type (code, markdown, text)
3. Generates embeddings using sentence-transformers
4. Builds and saves FAISS indices to data/index/
5. Saves ID mappings for chunk retrieval

v1.2 Updates:
- Added HNSW index support for faster approximate nearest neighbor search
- Configurable via FAISS_INDEX_TYPE setting ("flat" or "hnsw")
- HNSW parameters: M (connections), efConstruction (build accuracy)

Usage:
    python scripts/data_pipeline/indexing/build_faiss_index.py

Output:
    - data/index/code_faiss.index
    - data/index/markdown_faiss.index
    - data/index/text_faiss.index
    - data/index/{source_type}_id_mapping.json (for each type)

Author: Hay Hoffman
"""

import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.chunk import BaseChunk, create_chunk_from_dict
from settings import (
    ALL_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_EMBEDDING_DIM,
    FAISS_HNSW_EF_CONSTRUCTION,
    FAISS_HNSW_M,
    INDEX_DIR,
)

logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    """Load chunks from JSON file.

    Returns:
        list of chunk dictionaries

    Raises:
        FileNotFoundError: If chunks file doesn't exist
    """
    if not ALL_CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {ALL_CHUNKS_FILE}")

    logger.info(f"Loading chunks from {ALL_CHUNKS_FILE}")
    with open(ALL_CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} raw chunks")
    return chunks


def group_chunks_by_type(chunks_data: list[dict]) -> dict[str, list[BaseChunk]]:
    """Group chunks by source_type.

    Args:
        chunks_data: list of chunk dictionaries

    Returns:
        Dict mapping source_type to list of BaseChunk objects
    """
    grouped: dict[str, list[BaseChunk]] = {
        "code": [],
        "markdown": [],
        "text": [],
    }

    for chunk_data in chunks_data:
        try:
            chunk = create_chunk_from_dict(chunk_data)
            if chunk.source_type in grouped:
                grouped[chunk.source_type].append(chunk)
            else:
                logger.warning(f"Unknown source_type: {chunk.source_type}")
        except Exception as e:
            logger.warning(f"Failed to parse chunk: {e}")

    for source_type, chunks in grouped.items():
        logger.info(f"  {source_type}: {len(chunks)} chunks")

    return grouped


def generate_embeddings(
    chunks: list[BaseChunk],
    model: SentenceTransformer,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for chunks.

    Args:
        chunks: list of BaseChunk objects
        model: SentenceTransformer model
        batch_size: Batch size for encoding

    Returns:
        Numpy array of embeddings (n_chunks, embedding_dim)
    """
    # Extract content from chunks
    texts = [chunk.content for chunk in chunks]

    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    # Generate embeddings in batches with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
    )

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS HNSW index from embeddings.

    Uses HNSW (Hierarchical Navigable Small World) for fast approximate
    nearest neighbor search.

    Args:
        embeddings: Numpy array of embeddings (n_chunks, embedding_dim)

    Returns:
        FAISS HNSW index
    """
    dimension = embeddings.shape[1]

    # Build HNSW index for approximate nearest neighbor search
    # M: number of connections per layer (higher = more accurate but more memory)
    # efConstruction: depth of search during construction (higher = slower build, more accurate)
    index = faiss.IndexHNSWFlat(dimension, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = FAISS_HNSW_EF_CONSTRUCTION

    # Add vectors to index
    index.add(embeddings.astype("float32"))

    logger.info(
        f"Built HNSW index with {index.ntotal} vectors, dim={index.d}, "
        f"M={FAISS_HNSW_M}, efConstruction={FAISS_HNSW_EF_CONSTRUCTION}"
    )

    return index


def save_index_and_mapping(
    index: faiss.Index,
    chunks: list[BaseChunk],
    source_type: str,
    output_dir: Path,
) -> None:
    """Save FAISS index and ID mapping to disk.

    Args:
        index: FAISS index
        chunks: list of BaseChunk objects (in same order as index)
        source_type: One of "code", "markdown", "text"
        output_dir: Directory to save files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    index_path = output_dir / f"{source_type}_faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")

    # Build and save ID mapping (position -> chunk_id)
    id_mapping = {i: chunk.id for i, chunk in enumerate(chunks)}
    mapping_path = output_dir / f"{source_type}_id_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2)
    logger.info(f"Saved ID mapping to {mapping_path}")


def build_all_faiss_indices() -> int:
    """Build FAISS indices for all source types.

    Returns:
        0 on success, 1 on failure
    """
    try:
        # 1. Load chunks
        chunks_data = load_chunks()

        # 2. Group by type
        grouped_chunks = group_chunks_by_type(chunks_data)

        # 3. Load embedding model once
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Verify embedding dimension
        test_embedding = model.encode(["test"], convert_to_numpy=True)
        actual_dim = test_embedding.shape[1]
        if actual_dim != FAISS_EMBEDDING_DIM:
            logger.warning(
                f"Embedding dimension mismatch: expected {FAISS_EMBEDDING_DIM}, "
                f"got {actual_dim}. Update FAISS_EMBEDDING_DIM in const.py"
            )

        # 4. Build index for each type
        for source_type, chunks in grouped_chunks.items():
            if not chunks:
                logger.info(f"No chunks for {source_type}, skipping index build")
                continue

            logger.info(f"\n{'='*50}")
            logger.info(f"Building FAISS index for {source_type} ({len(chunks)} chunks)")
            logger.info(f"{'='*50}")

            # Generate embeddings
            embeddings = generate_embeddings(chunks, model)

            # Build FAISS index
            index = build_faiss_index(embeddings)

            # Save index and mapping
            save_index_and_mapping(index, chunks, source_type, INDEX_DIR)

            logger.info(f"[SUCCESS] Completed {source_type} FAISS index")

        # 5. Summary
        logger.info(f"\n{'='*50}")
        logger.info("FAISS INDEX BUILD SUMMARY")
        logger.info(f"{'='*50}")
        for source_type in ["code", "markdown", "text"]:
            index_path = INDEX_DIR / f"{source_type}_faiss.index"
            if index_path.exists():
                # Load to get stats
                index = faiss.read_index(str(index_path))
                logger.info(
                    f"  {source_type}: {index.ntotal} vectors, "
                    f"index={index_path.stat().st_size / 1024:.1f}KB"
                )
            else:
                logger.info(f"  {source_type}: No index (no chunks)")

        return 0

    except Exception as e:
        logger.error(f"FAISS index building failed: {e}", exc_info=True)
        return 1


def build_faiss() -> int:
    """Build FAISS indices from chunks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting FAISS index build...")
    result = build_all_faiss_indices()
    if result == 0:
        logger.info("[SUCCESS] FAISS index build complete!")
    return result


if __name__ == "__main__":
    sys.exit(build_faiss())
