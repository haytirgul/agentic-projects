"""Build BM25 index from code chunks for keyword-based retrieval.

This script creates a BM25 index from the code chunks JSON file,
which will be used for sparse retrieval in the hybrid search system.

Usage:
    python scripts/data_pipeline/indexing/build_bm25_index.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from settings import DATA_DIR, RAG_COMPONENTS_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INPUT_FILE = DATA_DIR / "processed" / "code_chunks.json"
BM25_OUTPUT = RAG_COMPONENTS_DIR / "bm25_index.pkl"
CHUNKS_OUTPUT = RAG_COMPONENTS_DIR / "chunks.pkl"


def tokenize_code(text: str) -> List[str]:
    """Tokenize code for BM25 indexing.

    Args:
        text: Code text

    Returns:
        List of tokens
    """
    # Simple tokenization: split on whitespace and special characters
    # Keep underscores for Python identifiers
    tokens = []
    current_token = []

    for char in text:
        if char.isalnum() or char == '_':
            current_token.append(char)
        else:
            if current_token:
                tokens.append(''.join(current_token).lower())
                current_token = []

    if current_token:
        tokens.append(''.join(current_token).lower())

    return tokens


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


def build_bm25_index(chunks: List[Dict[str, Any]]):
    """Build BM25 index from code chunks.

    Args:
        chunks: List of chunk dictionaries
    """
    logger.info("Building BM25 index...")

    # Tokenize all chunks
    tokenized_corpus = []
    for chunk in tqdm(chunks, desc="Tokenizing chunks"):
        # Combine all searchable text
        text_parts = [
            chunk['name'],
            chunk['full_name'],
            chunk['docstring'],
            chunk['content']
        ]
        text = " ".join(text_parts)

        tokens = tokenize_code(text)
        tokenized_corpus.append(tokens)

    # Build BM25 index
    logger.info("Creating BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index
    RAG_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BM25_OUTPUT, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index to {BM25_OUTPUT}")

    # Save chunks for retrieval
    with open(CHUNKS_OUTPUT, 'wb') as f:
        pickle.dump(chunks, f)
    logger.info(f"Saved chunks to {CHUNKS_OUTPUT}")

    logger.info(f"✅ BM25 index built with {len(chunks)} chunks")


def main():
    """Main entry point."""
    logger.info("Starting BM25 index build...")

    # Load chunks
    chunks = load_chunks()

    # Build index
    build_bm25_index(chunks)

    logger.info("✅ BM25 index build complete!")


if __name__ == "__main__":
    main()
