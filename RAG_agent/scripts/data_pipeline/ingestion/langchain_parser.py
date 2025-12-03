"""Script to parse langchain_llms_full.txt into individual markdown files."""

import re
import sys
from pathlib import Path
from typing import Tuple

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from settings import INPUT_DIR, OUTPUT_DIR


def extract_filename_from_url(url: str) -> str:
    """
    Extract filename from LangChain documentation URL.

    Args:
        url: URL like "https://docs.langchain.com/langsmith/add-auth-server"

    Returns:
        Relative path like "langsmith/add-auth-server.md"
    """
    # Remove the base URL
    path = url.replace("https://docs.langchain.com/", "").strip()
    # Ensure it ends with .md
    if not path.endswith(".md"):
        path = f"{path}.md"
    return path


def remove_xml_html_tags(content: str) -> str:
    """
    Remove all XML/HTML-like tags from content.

    Removes all tags matching:
    - Self-closing tags: <Tag ... />
    - Paired tags: <Tag>...</Tag>
    - Tags with attributes: <Tag attr="value">...</Tag>

    Uses iterative approach to handle nested tags properly.

    Args:
        content: Content string that may contain XML/HTML tags.

    Returns:
        Content with all XML/HTML tags removed.
    """
    # Pattern for self-closing tags
    self_closing_pattern = r'<[^>]+/>'
    content = re.sub(self_closing_pattern, '', content)

    # Pattern for paired tags (iterative removal for nested tags)
    while True:
        # Match opening and closing tags, including nested content
        paired_pattern = r'<[^>]+>(.*?)</[^>]+>'
        new_content = re.sub(paired_pattern, r'\1', content, flags=re.DOTALL)
        if new_content == content:
            # No more tags found
            break
        content = new_content

    return content.strip()


def parse_langchain_file(input_file: Path, output_dir: Path) -> Tuple[int, int]:
    """
    Parse LangChain llms.txt file into individual markdown files.

    Args:
        input_file: Path to the input llms.txt file
        output_dir: Directory to save parsed markdown files

    Returns:
        Tuple of (success_count, total_count)
    """
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 0, 0

    # Write to log file for debugging
    with open('parser_debug.log', 'w', encoding='utf-8') as log_file:
        log_file.write(f"Reading {input_file}...\n")
        content = input_file.read_text(encoding="utf-8")
        log_file.write(f"Content length: {len(content)}\n")

    # Split by document sections (looking for Source: lines)
    sections = []
    current_section = []
    lines = content.split('\n')

    for line in lines:
        if line.startswith('Source: https://docs.langchain.com/'):
            # Save previous section if it exists
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(line)

    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))

    with open('parser_debug.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"Found {len(sections)} document sections\n")

    success_count = 0
    total_count = len(sections)

    for i, section in enumerate(sections, 1):
        try:
            lines = section.split('\n')

            # Extract URL from the first Source: line
            url = None
            content_start = 0
            for j, line in enumerate(lines):
                if line.startswith('Source: https://docs.langchain.com/'):
                    # Extract URL from the line
                    url_match = re.search(r'Source: (https://docs\.langchain\.com/[^\s]+)', line)
                    if url_match:
                        url = url_match.group(1)
                    content_start = j + 1
                    break

            if not url:
                with open('parser_debug.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[{i}/{total_count}] Warning: No URL found, skipping\n")
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
            content = remove_xml_html_tags(content)

            # Skip if content is too short
            if len(content.strip()) < 50:
                with open('parser_debug.log', 'a', encoding='utf-8') as log_file:
                    log_file.write(f"[{i}/{total_count}] Warning: Content too short, skipping {filename}\n")
                continue

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            output_path.write_text(content, encoding="utf-8")
            with open('parser_debug.log', 'a', encoding='utf-8') as log_file:
                log_file.write(f"[{i}/{total_count}] [OK] Saved {filename}\n")
            success_count += 1

        except Exception as e:
            print(f"[{i}/{total_count}] Error processing section: {e}")
            continue

    return success_count, total_count


def main() -> int:
    """
    Parse langchain documentation file into individual markdown files.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    input_file = INPUT_DIR / "langchain_llms_full.txt"
    output_dir = OUTPUT_DIR / "md_files" / "langchain"

    with open('parser_debug.log', 'a', encoding='utf-8') as log_file:
        log_file.write(f"Parsing {input_file}...\n")
        log_file.write(f"Output directory: {output_dir}\n")

    success_count, total_count = parse_langchain_file(input_file, output_dir)

    if success_count == total_count:
        print(f"\n[OK] Successfully parsed {success_count} file(s)")
        return 0
    else:
        print(f"\n⚠ Parsed {success_count} out of {total_count} file(s)")
        return 1 if success_count == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
