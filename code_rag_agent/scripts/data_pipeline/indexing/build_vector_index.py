"""Build ChromaDB vector index from code chunks.

This script reads the code chunks JSON file, generates embeddings using
the configured embedding model, and stores them in ChromaDB for semantic search.

Usage:
    python scripts/data_pipeline/indexing/build_vector_index.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from settings import (
    DATA_DIR,
    CHROMA_PERSIST_DIRECTORY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INPUT_FILE = DATA_DIR / "processed" / "code_chunks.json"


def load_chunks() -> List[Dict[str, Any]]:
    """Load code chunks from JSON file.

    Returns:
        List of chunk dictionaries
    """
    if not INPUT_FILE.exists():
        logger.error(f"Chunks file not found: {INPUT_FILE}")
        logger.error("Run chunk_code.py first")
        sys.exit(1)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} code chunks")
    return chunks


def prepare_text_for_embedding(chunk: Dict[str, Any]) -> str:
    """Prepare chunk text for embedding.

    Args:
        chunk: Chunk dictionary

    Returns:
        Text string for embedding
    """
    parts = []

    # Add context
    if chunk['parent_context']:
        parts.append(f"# {chunk['parent_context']}.{chunk['name']}")
    else:
        parts.append(f"# {chunk['name']}")

    # Add docstring
    if chunk['docstring']:
        parts.append(f'"""{chunk["docstring"]}"""')

    # Add code (truncate if very long)
    code = chunk['content']
    if len(code) > 2000:  # Limit for embedding model
        code = code[:2000] + "\n# ..."

    parts.append(code)

    return "\n".join(parts)


def build_index(chunks: List[Dict[str, Any]]):
    """Build ChromaDB vector index from chunks.

    Args:
        chunks: List of chunk dictionaries
    """
    logger.info("Initializing ChromaDB...")

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=COLLECTION_NAME)
        logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Code chunks from httpx repository"}
    )
    logger.info(f"Created collection: {COLLECTION_NAME}")

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded")

    # Prepare data for indexing
    logger.info("Preparing chunks for embedding...")
    ids = []
    documents = []
    metadatas = []
    embeddings_list = []

    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Generate ID
        chunk_id = f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
        ids.append(chunk_id)

        # Prepare text for embedding
        text = prepare_text_for_embedding(chunk)
        documents.append(text)

        # Prepare metadata
        metadata = {
            "file_path": chunk['file_path'],
            "start_line": chunk['start_line'],
            "end_line": chunk['end_line'],
            "chunk_type": chunk['chunk_type'],
            "name": chunk['name'],
            "full_name": chunk['full_name'],
            "parent_context": chunk['parent_context'],
        }
        metadatas.append(metadata)

    # Generate embeddings in batches
    logger.info(f"Generating embeddings for {len(documents)} chunks...")
    embeddings_list = []
    for i in tqdm(range(0, len(documents), EMBEDDING_BATCH_SIZE), desc="Embedding batches"):
        batch = documents[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings_list.extend(batch_embeddings.tolist())

    # Add to ChromaDB
    logger.info("Adding embeddings to ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )

    logger.info(f"✅ Successfully indexed {len(chunks)} code chunks")
    logger.info(f"Vector database location: {CHROMA_PERSIST_DIRECTORY}")


def main():
    """Main entry point."""
    logger.info("Starting vector index build...")

    # Load chunks
    chunks = load_chunks()

    # Build index
    build_index(chunks)

    logger.info("✅ Vector index build complete!")


if __name__ == "__main__":
    main()
