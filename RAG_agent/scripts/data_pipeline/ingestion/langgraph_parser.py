"""Script to parse langgraph_llms_full.txt into individual markdown files."""

import re
import sys
from pathlib import Path
from typing import Tuple

# Note: Removed Windows console encoding setup to avoid I/O errors when imported as module
# All Unicode symbols have been replaced with ASCII equivalents

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import INPUT_DIR, OUTPUT_DIR


def remove_search_metadata(content: str) -> str:
    """
    Remove search metadata blocks from content.

    Removes blocks matching:
    ---
    search:
      boost: 2
    ---

    Args:
        content: Content string that may contain metadata blocks.

    Returns:
        Content with metadata blocks removed.
    """
    # Pattern to match metadata blocks
    pattern = r"^---\nsearch:\n\s+boost:\s+\d+\n---\n"
    # Remove all occurrences (multiline, case insensitive)
    cleaned = re.sub(pattern, "", content, flags=re.MULTILINE | re.IGNORECASE)
    return cleaned.strip()


def extract_filename_from_path(path: str) -> str:
    """
    Extract and normalize filename from path.

    Args:
        path: Relative path like "how-tos/use-remote-graph.md"

    Returns:
        Normalized relative path like "how-tos/use-remote-graph.md"
    """
    # Strip whitespace
    path = path.strip()
    # Ensure it ends with .md
    if not path.endswith(".md"):
        path = f"{path}.md"
    return path


def parse_langgraph_file(input_file: Path, output_dir: Path) -> Tuple[int, int]:
    """
    Parse LangGraph llms.txt file into individual markdown files.

    The file format is:
    ---
    path/to/file.md
    ---

    # Title
    content...

    ---
    another/file.md
    ---
    ...

    Args:
        input_file: Path to the input llms.txt file
        output_dir: Directory to save parsed markdown files

    Returns:
        Tuple of (success_count, total_count)
    """
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 0, 0

    print(f"Reading {input_file}...")
    content = input_file.read_text(encoding="utf-8")

    # Split by document sections using the pattern: ---\nfilename.md\n---
    # Pattern: ---\n followed by filename, followed by \n---\n, then content
    pattern = r'---\n([^\n]+\.md)\n---\n'

    # Find all section headers
    matches = list(re.finditer(pattern, content))

    print(f"Found {len(matches)} document sections")

    success_count = 0
    total_count = len(matches)

    for i, match in enumerate(matches):
        try:
            # Get the filename from the match
            filename = match.group(1).strip()

            # Get the content start position (after the header)
            content_start = match.end()

            # Get the content end position (before next header or end of file)
            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(content)

            # Extract the content
            section_content = content[content_start:content_end].strip()

            # Clean the content
            section_content = remove_search_metadata(section_content)

            # Skip if content is too short
            if len(section_content.strip()) < 50:
                print(f"[{i+1}/{total_count}] Warning: Content too short, skipping {filename}")
                continue

            # Normalize filename
            filename = extract_filename_from_path(filename)
            output_path = output_dir / filename

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            output_path.write_text(section_content, encoding="utf-8")
            print(f"[{i+1}/{total_count}] [OK] Saved {filename}")
            success_count += 1

        except Exception as e:
            print(f"[{i+1}/{total_count}] Error processing section: {e}")
            import traceback
            traceback.print_exc()
            continue

    return success_count, total_count


def main() -> int:
    """
    Parse langgraph documentation file into individual markdown files.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    input_file = INPUT_DIR / "langgraph_llms_full.txt"
    output_dir = OUTPUT_DIR / "md_files" / "langgraph"

    print(f"Parsing {input_file}...")
    print(f"Output directory: {output_dir}\n")

    success_count, total_count = parse_langgraph_file(input_file, output_dir)

    if success_count == total_count:
        print(f"\n[OK] Successfully parsed {success_count} file(s)")
        return 0
    else:
        print(f"\n[WARNING] Parsed {success_count} out of {total_count} file(s)")
        return 1 if success_count == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
