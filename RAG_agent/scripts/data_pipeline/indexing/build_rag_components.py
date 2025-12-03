"""
Build and save RAG components as pickle files.

This script builds the DocumentIndex, BM25, and Fuzzy matcher components
and saves them as pickle files for fast loading during runtime.

This should be run ONCE after setting up the knowledge base.

Usage:
    python scripts/data_pipeline/indexing/build_rag_components.py

Output:
    - data/rag_components/document_index.pkl
    - data/rag_components/bm25_index.pkl
"""

import sys
import logging
import pickle
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import DATA_DIR  # noqa: E402
from src.rag.document_index import DocumentIndex  # noqa: E402
from src.rag.bm25_search import BM25SearchEngine  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Build and save RAG components."""
    logger.info("=" * 80)
    logger.info("RAG COMPONENTS BUILD SCRIPT")
    logger.info("=" * 80)

    # Define paths
    knowledge_base_path = DATA_DIR / "output" / "json_files"
    output_dir = DATA_DIR / "rag_components"

    logger.info(f"Knowledge base path: {knowledge_base_path}")
    logger.info(f"Output directory: {output_dir}")

    # Check if knowledge base exists
    if not knowledge_base_path.exists():
        logger.error(f"Knowledge base not found at {knowledge_base_path}")
        logger.error("Please run data ingestion first:")
        logger.error("  python scripts/data_pipeline/processing/build_rag_index.py")
        return 1

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Build DocumentIndex
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Building DocumentIndex")
    logger.info("=" * 80)

    try:
        logger.info("Loading and parsing JSON documents...")
        document_index = DocumentIndex(knowledge_base_path)
        all_docs = document_index.get_all_documents()
        logger.info(f"[OK] Loaded {len(all_docs)} documents")

        # Save to pickle
        index_path = output_dir / "document_index.pkl"
        logger.info(f"Saving DocumentIndex to {index_path}...")
        with open(index_path, 'wb') as f:
            pickle.dump(document_index, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = index_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] DocumentIndex saved ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Failed to build DocumentIndex: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 2: Build BM25 Index
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Building BM25 Index")
    logger.info("=" * 80)

    try:
        logger.info("Building BM25 keyword search index...")
        bm25_engine = BM25SearchEngine(all_docs)
        logger.info(f"[OK] Built BM25 index over {len(all_docs)} documents")

        # Save to pickle
        bm25_path = output_dir / "bm25_index.pkl"
        logger.info(f"Saving BM25 index to {bm25_path}...")
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_engine, f, protocol=pickle.HIGHEST_PROTOCOL)

        file_size_mb = bm25_path.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] BM25 index saved ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Failed to build BM25 index: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("RAG COMPONENTS BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Components saved to: {output_dir}")
    logger.info("Files:")
    logger.info("  - document_index.pkl")
    logger.info("  - bm25_index.pkl")

    return 0


if __name__ == "__main__":
    sys.exit(main())
