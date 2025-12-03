"""Convert markdown files to hierarchical JSON structures.

This script converts all markdown documentation files to structured JSON format
for use by the RAG system components.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.data_pipeline.processing.markdown_to_json import convert_all_markdown_files
from settings import OUTPUT_DIR


def main() -> None:
    """Convert markdown files to JSON."""
    print("=" * 60)
    print("Markdown to JSON Conversion")
    print("=" * 60)

    input_dir = OUTPUT_DIR / "md_files"
    json_dir = OUTPUT_DIR / "json_files"

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {json_dir}")
    print(f"Input dir exists: {input_dir.exists()}")
    print("-" * 60)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # List markdown files
    md_files = list(input_dir.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files")
    for f in md_files[:5]:  # Show first 5
        print(f"  {f}")

    json_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created JSON directory: {json_dir}")

    success_count, total_count = convert_all_markdown_files(input_dir, json_dir)
    print(f"Conversion result: {success_count}/{total_count}")

    if success_count == 0:
        print("No files converted. Exiting.")
        return

    print(f"\n[OK] Successfully converted {success_count}/{total_count} markdown files to JSON")
    print(f"JSON files saved to: {json_dir}")
    print("\nYou can now build RAG components from these JSON files.")


if __name__ == "__main__":
    main()
