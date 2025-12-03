"""
Build script to create ChromaDB vector index with content-block-level granularity.

This script loads all documents from the knowledge base and builds a ChromaDB
vector database with HNSW indexing for fast similarity search.

Usage:
    python scripts/data_pipeline/indexing/build_vector_index.py [--force-rebuild]

Options:
    --force-rebuild  Delete existing collection and rebuild from scratch

Expected time: 30-90 seconds for ~10,000-15,000 embeddings (with content blocks)
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to Python path (must be before local imports)
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import DATA_DIR, EMBEDDING_MODEL, GOOGLE_API_KEY  # noqa: E402
from src.rag.document_index import DocumentIndex  # noqa: E402
from src.rag.vector_search import VectorSearchEngine  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(force_rebuild: bool = False, disable_blocks: bool = False):
    """Build vector database index with content-block-level granularity."""
    logger.info("=" * 80)
    logger.info("VECTOR DATABASE INDEX BUILD SCRIPT")
    logger.info("=" * 80)
    logger.info(f"Force rebuild: {force_rebuild}")
    logger.info("")

    # Validate API key if using Google
    if EMBEDDING_MODEL.startswith("models/"):
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY not set in environment")
            logger.error("Please set it in your .env file")
            return 1

    # Define paths
    knowledge_base_path = DATA_DIR / "output" / "json_files"
    vector_db_path = DATA_DIR / "vector_db"

    logger.info(f"Knowledge base path: {knowledge_base_path}")
    logger.info(f"Vector DB path: {vector_db_path}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")

    # Check if knowledge base exists
    if not knowledge_base_path.exists():
        logger.error(f"Knowledge base not found at {knowledge_base_path}")
        logger.error("Please run data ingestion first:")
        logger.error("  python scripts/data_pipeline/processing/build_rag_index.py")
        return 1

    # Load documents
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading documents")
    logger.info("=" * 80)

    try:
        index = DocumentIndex(knowledge_base_path)
        all_docs = index.get_all_documents()
        logger.info(f"Loaded {len(all_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Initialize vector search engine
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Initializing ChromaDB")
    logger.info("=" * 80)

    try:
        collection_name = "documentation_embeddings"
        enable_content_blocks = not disable_blocks

        # Create vector search engine (this handles ChromaDB initialization)
        vector_engine = VectorSearchEngine(
            collection_name=collection_name,
            embedding_model=EMBEDDING_MODEL,
            persist_directory=str(vector_db_path),
            enable_block_level=enable_content_blocks
        )

        # If force rebuild, reset the collection
        if force_rebuild:
            try:
                vector_engine.reset()
                logger.info("Reset existing collection")
            except Exception:
                logger.info("No existing collection to reset")

        # Test the engine
        test_query = "test query"
        test_results = vector_engine.search(test_query, top_k=1)
        logger.info("[OK] ChromaDB initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Build embeddings
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Building Embeddings")
    logger.info("=" * 80)

    try:
        logger.info(f"Processing {len(all_docs)} documents...")
        logger.info("This may take 30-90 seconds depending on document count...")

        # Build index with progress tracking
        total_processed = vector_engine.build_index(all_docs)

        logger.info(f"[OK] Built embeddings for {total_processed} content blocks")

        # Final validation
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Validation")
        logger.info("=" * 80)

        # Test search
        test_results = vector_engine.search("LangGraph", top_k=3)
        if len(test_results) > 0:
            logger.info(f"[OK] Search test passed: Found {len(test_results)} results")
        else:
            logger.warning("⚠ Search test returned no results")

        # Get final count
        final_count = vector_engine.get_collection_count()
        logger.info(f"[OK] Final collection size: {final_count} embeddings")

    except Exception as e:
        logger.error(f"Failed to build embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("VECTOR DATABASE BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Embeddings: {final_count}")
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Content blocks: {enable_content_blocks}")

    return 0


if __name__ == "__main__":
    # Parse command line arguments for backward compatibility
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild by deleting existing collection"
    )
    parser.add_argument(
        "--disable-blocks",
        action="store_true",
        help="Disable content-block-level embeddings (only doc+section)"
    )
    args = parser.parse_args()

    sys.exit(main(force_rebuild=args.force_rebuild, disable_blocks=args.disable_blocks))
