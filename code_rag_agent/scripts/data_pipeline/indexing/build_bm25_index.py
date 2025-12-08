"""Build BM25 index from chunks for keyword-based retrieval.

This script creates a BM25 index from the all_chunks.json file,
using code-aware tokenization as specified in v1.1 architecture.

Usage:
    python scripts/data_pipeline/indexing/build_bm25_index.py

Output:
    - data/rag_components/bm25_index.pkl
    - data/rag_components/bm25_corpus.pkl (tokenized corpus)
    - data/rag_components/bm25_chunk_ids.json (position -> chunk_id mapping)

Author: Hay Hoffman
"""

import json
import logging
import pickle
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from models.chunk import create_chunk_from_dict
from settings import (  
    ALL_CHUNKS_FILE,
    BM25_KEEP_HEX_CODES,
    BM25_MIN_TOKEN_LENGTH,
    BM25_SPLIT_CAMELCASE,
    BM25_SPLIT_SNAKE_CASE,
    BM25_STOPWORDS,
    RAG_COMPONENTS_DIR,
)

logger = logging.getLogger(__name__)

# Output paths
BM25_INDEX_FILE = RAG_COMPONENTS_DIR / "bm25_index.pkl"
BM25_CORPUS_FILE = RAG_COMPONENTS_DIR / "bm25_corpus.pkl"
BM25_CHUNK_IDS_FILE = RAG_COMPONENTS_DIR / "bm25_chunk_ids.json"


def tokenize_code_aware(text: str) -> list[str]:
    """Tokenize text with code-aware rules (v1.1).

    Handles:
    - camelCase splitting (HTTPClient → http, client)
    - snake_case splitting (get_user → get, user)
    - Hex code preservation (0x884 → 0x884)
    - Minimum token length filtering
    - Stopword removal

    Args:
        text: Text to tokenize

    Returns:
        list of lowercase tokens
    """
    tokens = []

    # First, extract and preserve hex codes
    hex_pattern = r'0x[0-9A-Fa-f]+'
    hex_codes = re.findall(hex_pattern, text)
    if BM25_KEEP_HEX_CODES:
        tokens.extend([hc.lower() for hc in hex_codes])
        # Remove hex codes from text for further processing
        text = re.sub(hex_pattern, ' ', text)

    # Split on non-alphanumeric (except underscore)
    words = re.split(r'[^a-zA-Z0-9_]+', text)

    for word in words:
        if not word:
            continue

        # Split camelCase: HTTPClient → HTTP, Client → http, client
        if BM25_SPLIT_CAMELCASE:
            # Handle consecutive uppercase (HTTP) followed by lowercase
            camel_parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', word)
            # Handle lowercase followed by uppercase
            camel_parts = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', camel_parts)
            word_parts = camel_parts.split()
        else:
            word_parts = [word]

        # Process each part
        for part in word_parts:
            # Split snake_case
            if BM25_SPLIT_SNAKE_CASE and '_' in part:
                snake_parts = part.split('_')
            else:
                snake_parts = [part]

            for token in snake_parts:
                token_lower = token.lower()

                # Skip if too short (unless it's a hex code)
                if len(token_lower) < BM25_MIN_TOKEN_LENGTH:
                    continue

                # Skip stopwords
                if token_lower in BM25_STOPWORDS:
                    continue

                tokens.append(token_lower)

    return tokens


def load_chunks() -> list[dict]:
    """Load chunks from JSON file.

    Returns:
        list of chunk dictionaries
    """
    if not ALL_CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {ALL_CHUNKS_FILE}")

    logger.info(f"Loading chunks from {ALL_CHUNKS_FILE}")
    with open(ALL_CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks


def build_bm25_index(chunks_data: list[dict]) -> int:
    """Build BM25 index from chunks.

    Args:
        chunks_data: list of chunk dictionaries

    Returns:
        0 on success, 1 on failure
    """
    logger.info("Building BM25 index with code-aware tokenization...")

    # Create output directory
    RAG_COMPONENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenize all chunks
    tokenized_corpus = []
    chunk_ids = []
    skipped = 0

    for chunk_data in chunks_data:
        try:
            chunk = create_chunk_from_dict(chunk_data)

            # Combine searchable text
            text_parts = [
                chunk.name,
                chunk.full_name,
                chunk.content,
            ]

            # Add docstring for code chunks
            if chunk.docstring:
                text_parts.append(chunk.docstring)

            text = " ".join(filter(None, text_parts))
            tokens = tokenize_code_aware(text)

            if tokens:
                tokenized_corpus.append(tokens)
                chunk_ids.append(chunk.id)
            else:
                skipped += 1

        except Exception as e:
            logger.warning(f"Failed to process chunk: {e}")
            skipped += 1

    logger.info(f"Tokenized {len(tokenized_corpus)} chunks ({skipped} skipped)")

    # Build BM25 index
    logger.info("Creating BM25Okapi index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Save BM25 index
    with open(BM25_INDEX_FILE, 'wb') as f:
        pickle.dump(bm25, f)
    logger.info(f"Saved BM25 index to {BM25_INDEX_FILE}")

    # Save tokenized corpus (for debugging/analysis)
    with open(BM25_CORPUS_FILE, 'wb') as f:
        pickle.dump(tokenized_corpus, f)
    logger.info(f"Saved tokenized corpus to {BM25_CORPUS_FILE}")

    # Save chunk IDs mapping (position -> chunk_id)
    with open(BM25_CHUNK_IDS_FILE, 'w', encoding='utf-8') as f:
        json.dump({i: cid for i, cid in enumerate(chunk_ids)}, f, indent=2)
    logger.info(f"Saved chunk IDs to {BM25_CHUNK_IDS_FILE}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("BM25 INDEX BUILD SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"  Total chunks indexed: {len(chunk_ids)}")
    logger.info(f"  Average tokens per chunk: {sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus):.1f}")
    logger.info(f"  Index file size: {BM25_INDEX_FILE.stat().st_size / 1024:.1f}KB")

    return 0


def build_bm25() -> int:
    """Build BM25 index from chunks."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting BM25 index build...")

    try:
        # Load chunks
        chunks_data = load_chunks()

        # Build index
        result = build_bm25_index(chunks_data)

        if result == 0:
            logger.info("[SUCCESS] BM25 index build complete!")

        return result

    except Exception as e:
        logger.error(f"BM25 index building failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_bm25())
