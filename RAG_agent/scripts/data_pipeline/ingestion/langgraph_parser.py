"""Script to parse langgraph_llms_full.txt into individual markdown files."""

import re
import sys
from pathlib import Path
from typing import List, Tuple

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


def extract_filename_from_url(url: str) -> str:
    """
    Extract filename from LangGraph documentation URL.

    Args:
        url: URL like "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts"

    Returns:
        Relative path like "concepts/agentic_concepts.md"
    """
    # Remove the base URL
    path = url.replace("https://langchain-ai.github.io/langgraph/", "").strip()
    # Ensure it ends with .md
    if not path.endswith(".md"):
        path = f"{path}.md"
    return path


def parse_langgraph_file(input_file: Path, output_dir: Path) -> Tuple[int, int]:
    """
    Parse LangGraph llms.txt file into individual markdown files.

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

    # Split by document sections (looking for Source: lines)
    sections = []
    current_section = []
    lines = content.split('\n')

    for line in lines:
        if line.startswith('# ') and 'Source: https://langchain-ai.github.io/langgraph/' in line:
            # Save previous section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(line)

    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))

    print(f"Found {len(sections)} document sections")

    success_count = 0
    total_count = len(sections)

    for i, section in enumerate(sections, 1):
        try:
            lines = section.split('\n')

            # Extract URL from the first Source: line
            url = None
            content_start = 0
            for j, line in enumerate(lines):
                if line.startswith('# ') and 'Source: https://langchain-ai.github.io/langgraph/' in line:
                    # Extract URL from the line
                    url_match = re.search(r'Source: (https://langchain-ai\.github\.io/langgraph/[^\s]+)', line)
                    if url_match:
                        url = url_match.group(1)
                    content_start = j + 1
                    break

            if not url:
                print(f"[{i}/{total_count}] Warning: No URL found, skipping")
                continue

            # Extract filename from URL
            filename = extract_filename_from_url(url)
            output_path = output_dir / filename

            # Get content after the header
            content_lines = lines[content_start:]

            # Remove the first few lines if they're empty or just formatting
            while content_lines and (content_lines[0].strip() == '' or content_lines[0].startswith('---')):
                content_lines = content_lines[1:]

            # Join content and clean it
            content = '\n'.join(content_lines)
            content = remove_search_metadata(content)

            # Skip if content is too short
            if len(content.strip()) < 50:
                print(f"[{i}/{total_count}] Warning: Content too short, skipping {filename}")
                continue

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            output_path.write_text(content, encoding="utf-8")
            print(f"[{i}/{total_count}] [OK] Saved {filename}")
            success_count += 1

        except Exception as e:
            print(f"[{i}/{total_count}] Error processing section: {e}")
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
        print(f"\n⚠ Parsed {success_count} out of {total_count} file(s)")
        return 1 if success_count == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
