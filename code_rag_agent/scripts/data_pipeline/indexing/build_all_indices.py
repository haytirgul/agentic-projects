"""Build all indices for the code RAG system.

This script orchestrates the complete data pipeline:
1. Clone/update httpx repository
2. Process all files into chunks
3. Build FAISS vector indices
4. Build BM25 keyword index

Usage:
    python scripts/data_pipeline/indexing/build_all_indices.py

Author: Hay Hoffman
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline functions
from scripts.data_pipeline.ingestion.clone_httpx_repo import clone_or_update_repo
from scripts.data_pipeline.processing.process_all_files import main as process_files
from scripts.data_pipeline.indexing.build_bm25_index import build_bm25
from scripts.data_pipeline.indexing.build_faiss_index import build_faiss

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the complete data pipeline.

    Returns:
        0 on success, 1 on failure
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("  CODE RAG AGENT - DATA PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Clone/update repository
    print("[1/4] Cloning/updating httpx repository...")
    if not clone_or_update_repo():
        print("[ERROR] Clone failed")
        return 1
    print("[OK] Repository ready")
    print()

    # Step 2: Process files into chunks
    print("[2/4] Processing files into chunks...")
    if process_files() != 0:
        print("[ERROR] Processing failed")
        return 1
    print("[OK] Files processed")
    print()

    # Step 3: Build FAISS indices
    print("[3/4] Building FAISS vector indices...")
    if build_faiss() != 0:
        print("[ERROR] FAISS build failed")
        return 1
    print("[OK] FAISS indices built")
    print()

    # Step 4: Build BM25 index
    print("[4/4] Building BM25 keyword index...")
    if build_bm25() != 0:
        print("[ERROR] BM25 build failed")
        return 1
    print("[OK] BM25 index built")
    print()

    print("=" * 60)
    print("  [SUCCESS] All indices built successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
