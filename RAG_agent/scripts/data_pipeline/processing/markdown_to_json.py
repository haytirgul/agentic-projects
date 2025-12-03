"""Convert markdown files to hierarchical JSON structures.

This module parses markdown files, extracts heading hierarchy, content, and links,
then outputs hierarchical JSON files preserving the document structure.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.document import CodeContentBlock, DocumentSection, Link, TableContentBlock, TextBlock


def generate_link_id(text: str) -> str:
    """Generate a normalized identifier from link text for LLM search.

    Args:
        text: Link text

    Returns:
        Normalized identifier (lowercase, spaces to underscores, special chars removed)
    """
    # Convert to lowercase
    identifier = text.lower()
    # Replace spaces and common separators with underscores
    identifier = re.sub(r'[\s\-_]+', '_', identifier)
    # Remove special characters, keep alphanumeric and underscores
    identifier = re.sub(r'[^a-z0-9_]', '', identifier)
    # Remove multiple consecutive underscores
    identifier = re.sub(r'_+', '_', identifier)
    # Remove leading/trailing underscores
    identifier = identifier.strip('_')
    # Ensure it's not empty
    if not identifier:
        identifier = "link"
    return identifier


def extract_links(content: str) -> List[Link]:
    """Extract all markdown links and inline URLs from content with identifiers.

    Only extracts links that start with https:// (excludes relative links, anchors, http).

    Args:
        content: Markdown content string

    Returns:
        List of Link objects with identifiers
    """
    links = []

    # Pattern for markdown links: [text](url)
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(markdown_pattern, content):
        text = match.group(1).strip()
        url = match.group(2).strip()

        # Only include HTTPS links (exclude relative and HTTP links)
        if url.startswith('https://'):
            link_id = generate_link_id(text)
            links.append(Link(id=link_id, url=url))

    # Pattern for inline URLs (not in markdown format)
    url_pattern = r'https://[^\s<>"\'{}|\\^`\[\]]+'
    for match in re.finditer(url_pattern, content):
        url = match.group(0).strip()

        # Skip if this URL is already captured as a markdown link
        if not any(link.url == url for link in links):
            # Generate ID from URL path
            path_part = url.replace('https://', '').split('/')[1] if '/' in url.replace('https://', '') else 'link'
            link_id = generate_link_id(path_part)
            links.append(Link(id=link_id, url=url))

    return links


def parse_content_blocks(content: str) -> List:
    """Parse content into content blocks (text and code blocks).

    Args:
        content: Raw content string

    Returns:
        List of ContentBlock objects (TextBlock or CodeContentBlock)
    """
    blocks = []
    lines = content.split('\n')
    i = 0
    current_text = []

    while i < len(lines):
        line = lines[i]

        # Check for code block start
        code_fence_match = re.match(r'^```(\w+)?', line)
        if code_fence_match:
            # Save any accumulated text first
            if current_text:
                text_content = '\n'.join(current_text).strip()
                if text_content:
                    blocks.append(TextBlock(type="text", content=text_content))
                current_text = []

            # Extract code block
            language = code_fence_match.group(1) if code_fence_match.group(1) else None
            code_lines = []
            i += 1

            # Find closing fence
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1

            code_content = '\n'.join(code_lines)
            if code_content.strip():
                blocks.append(CodeContentBlock(type="code", content=code_content, header=None))

        else:
            # Regular text line
            current_text.append(line)

        i += 1

    # Add any remaining text
    if current_text:
        text_content = '\n'.join(current_text).strip()
        if text_content:
            blocks.append(TextBlock(type="text", content=text_content))

    return blocks


def parse_markdown_content(content: str) -> DocumentSection:
    """Parse markdown content into a hierarchical document structure.

    Args:
        content: Raw markdown content

    Returns:
        DocumentSection representing the parsed content
    """
    lines = content.split('\n')
    sections = []
    current_section = None
    current_content = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for heading
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # Save previous section if exists
            if current_section:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    # Parse content into blocks
                    current_section.content_blocks = parse_content_blocks(content_text)
                    # Extract links from content
                    current_section.links = extract_links(content_text)
                sections.append(current_section)

            # Create new section
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            current_section = DocumentSection(
                title=title,
                level=level,
                links=[],
                content_blocks=None,
                children=[]
            )
            current_content = []

        elif current_section:
            # Add line to current section content
            current_content.append(line)

        i += 1

    # Add final section
    if current_section:
        content_text = '\n'.join(current_content).strip()
        if content_text:
            current_section.content_blocks = parse_content_blocks(content_text)
            current_section.links = extract_links(content_text)
        sections.append(current_section)

    # Build hierarchy
    root_section = DocumentSection(
        title="Root",
        level=0,
        links=[],
        content_blocks=None,
        children=[]
    )

    # Organize sections into hierarchy
    section_stack = [root_section]

    for section in sections:
        # Find the appropriate parent level
        while len(section_stack) > 1 and section_stack[-1].level >= section.level:
            section_stack.pop()

        # Add as child to current parent
        section_stack[-1].children.append(section)
        section_stack.append(section)

    return root_section


def convert_markdown_file(input_file: Path, output_file: Path) -> bool:
    """Convert a single markdown file to JSON.

    Args:
        input_file: Path to markdown file
        output_file: Path to output JSON file

    Returns:
        True if conversion successful
    """
    try:
        # Read markdown content
        content = input_file.read_text(encoding='utf-8')

        # Parse into document structure
        root_section = parse_markdown_content(content)

        # Convert to dict for JSON serialization using Pydantic's model_dump
        document_dict = root_section.model_dump(exclude_none=True)

        # Write JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(document_dict, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        return False


def convert_all_markdown_files(input_dir: Path, output_dir: Path) -> Tuple[int, int]:
    """Convert all markdown files in input directory to JSON.

    Args:
        input_dir: Directory containing markdown files
        output_dir: Output directory for JSON files

    Returns:
        Tuple of (success_count, total_count)
    """
    # Find all markdown files
    md_files = list(input_dir.rglob("*.md"))
    total_count = len(md_files)

    if total_count == 0:
        print(f"No markdown files found in {input_dir}")
        return 0, 0

    print(f"Found {total_count} markdown files to process...")

    success_count = 0
    for md_file in md_files:
        # Create corresponding JSON path
        relative_path = md_file.relative_to(input_dir)
        json_file = (output_dir / relative_path).with_suffix('.json')

        if convert_markdown_file(md_file, json_file):
            success_count += 1


    print(f"\nConverted {success_count}/{total_count} files")
    return success_count, total_count
