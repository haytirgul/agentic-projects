"""Unified file processing pipeline.

This script runs all individual processors and combines their outputs
into a single comprehensive dataset for RAG.

Usage:
    python scripts/data_pipeline/processing/process_all_files.py
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from settings import ALL_CHUNKS_FILE, DATA_DIR

# Setup project paths and imports
from utils import setup_project_paths

# Initialize project and logging
project_root = setup_project_paths()


logger = logging.getLogger(__name__)

PROCESSORS = [
    ("Python code", "chunk_code.py"),
    ("Markdown docs", "process_markdown.py"),
    ("Text files", "process_text_files.py"),
]


def run_processor(script_name: str, description: str) -> bool:
    """Run an individual processor script."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        logger.error(f"Processor script not found: {script_path}")
        return False

    logger.info(f"Running {description} processor...")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"✅ {description} processor completed")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} processor failed: {e.stderr}")
        return False


def combine_chunks() -> list[dict[str, Any]]:
    """Combine all chunk files into a single dataset."""
    all_chunks = []

    processed_dir = DATA_DIR / "processed"

    if not processed_dir.exists():
        logger.error(f"Processed directory not found: {processed_dir}")
        return []

    chunk_files = [
        ("code_chunks.json", "Python code"),
        ("markdown_chunks.json", "Markdown docs"),
        ("text_chunks.json", "Text files"),
    ]

    for filename, description in chunk_files:
        file_path = processed_dir / filename

        if not file_path.exists():
            logger.warning(f"Chunk file not found: {file_path}")
            continue

        try:
            with open(file_path, encoding='utf-8') as f:
                chunks = json.load(f)

            logger.info(f"Loaded {len(chunks)} {description} chunks from {filename}")
            all_chunks.extend(chunks)

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")

    return all_chunks


def main():
    """Main entry point."""
    logger.info("Starting unified file processing pipeline...")

    # Run all processors
    success_count = 0
    for description, script_name in PROCESSORS:
        if run_processor(script_name, description):
            success_count += 1
        else:
            logger.warning(f"Processor {description} failed, continuing...")

    if success_count == 0:
        logger.error("No processors completed successfully")
        sys.exit(1)

    logger.info(f"Completed {success_count}/{len(PROCESSORS)} processors")

    # Combine all chunks
    logger.info("Combining all chunks...")
    all_chunks = combine_chunks()

    if not all_chunks:
        logger.error("No chunks found to combine")
        sys.exit(1)

    logger.info(f"Total combined chunks: {len(all_chunks)}")

    # Save combined dataset
    ALL_CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALL_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved combined dataset to {ALL_CHUNKS_FILE}")

    # Statistics
    chunk_types = {}
    file_types = {}

    for chunk in all_chunks:
        # Count by chunk type
        chunk_type = chunk.get('chunk_type', 'unknown')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        # Count by file extension
        file_path = chunk.get('file_path', '')
        if '.' in file_path:
            ext = Path(file_path).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1

    logger.info("Combined chunk statistics:")
    logger.info("By chunk type:")
    for chunk_type, count in sorted(chunk_types.items()):
        logger.info(f"  {chunk_type}: {count}")

    logger.info("By file extension:")
    for ext, count in sorted(file_types.items()):
        logger.info(f"  {ext}: {count}")

    logger.info("🎉 File processing pipeline completed successfully!")


if __name__ == "__main__":
    main()
