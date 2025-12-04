"""Master script to build all indices for the code RAG system.

This script runs the complete data pipeline:
1. Clone/update httpx repository
2. Chunk code into semantic units
3. Build vector index (ChromaDB)
4. Build BM25 keyword index

Usage:
    python scripts/data_pipeline/build_all_indices.py
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_step(step_name: str, script_path: Path) -> bool:
    """Run a pipeline step.

    Args:
        step_name: Name of the step (for logging)
        script_path: Path to the script to run

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info(f"STEP: {step_name}")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        # Import and run the script's main function
        import importlib.util
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Run main function
        module.main()

        elapsed = time.time() - start_time
        logger.info(f"✅ {step_name} completed in {elapsed:.2f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ {step_name} failed after {elapsed:.2f}s: {e}")
        return False


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("CODE RAG AGENT - FULL PIPELINE BUILD")
    logger.info("=" * 80)

    pipeline_start = time.time()

    # Define pipeline steps
    steps = [
        ("Clone/Update httpx Repository", project_root / "scripts" / "data_pipeline" / "ingestion" / "clone_httpx_repo.py"),
        ("Chunk Code Files", project_root / "scripts" / "data_pipeline" / "processing" / "chunk_code.py"),
        ("Build Vector Index", project_root / "scripts" / "data_pipeline" / "indexing" / "build_vector_index.py"),
        ("Build BM25 Index", project_root / "scripts" / "data_pipeline" / "indexing" / "build_bm25_index.py"),
    ]

    # Run each step
    results = []
    for step_name, script_path in steps:
        success = run_step(step_name, script_path)
        results.append((step_name, success))

        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            logger.error("Aborting remaining steps")
            break

    # Summary
    pipeline_elapsed = time.time() - pipeline_start
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)

    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{status} - {step_name}")

    all_passed = all(success for _, success in results)

    logger.info("=" * 80)
    logger.info(f"Total time: {pipeline_elapsed:.2f}s ({pipeline_elapsed / 60:.2f}m)")

    if all_passed:
        logger.info("✅ ALL INDICES BUILT SUCCESSFULLY!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run: python app.py")
        logger.info("  2. Start asking questions about the httpx codebase!")
        sys.exit(0)
    else:
        logger.error("❌ PIPELINE FAILED")
        logger.error("Check the logs above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
