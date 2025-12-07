"""Structure-Aware Markdown Chunking Pipeline.

This script implements the PRD requirements for markdown processing:
1. Pass 1: Split by markdown headers using MarkdownHeaderTextSplitter
2. Pass 2: Size-constraint splitting with RecursiveCharacterTextSplitter when needed

Usage:
    python scripts/data_pipeline/processing/process_markdown.py
"""

import logging
import re
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Setup project paths first - must be done before importing from settings/models
from utils import setup_project_paths

project_root = setup_project_paths()

from models.chunk import MarkdownChunk
from settings import (
    CHUNK_OVERLAP_MARKDOWN,
    HTTPX_REPO_DIR,
    MARKDOWN_CHUNKS_FILE,
    MAX_CHUNK_SIZE_MARKDOWN,
    REPO_EXCLUDE_PATTERNS,
)
from utils import (
    create_chunk_id,
    estimate_token_count,
    generate_processing_stats,
    print_processing_stats,
    safe_read_file,
    save_json_chunks,
    should_exclude,
)

logger = logging.getLogger(__name__)


class StructureAwareMarkdownSplitter:
    """Structure-Aware Markdown Splitter compatible with LangChain/LlamaIndex interfaces.

    Implements the PRD two-pass hybrid chunking strategy:
    1. Pass 1: Use MarkdownHeaderTextSplitter (LangChain industry tool)
    2. Pass 2: Apply RecursiveCharacterTextSplitter when chunks exceed size limits
    """

    def __init__(
        self,
        chunk_size: int = MAX_CHUNK_SIZE_MARKDOWN,
        chunk_overlap: int = CHUNK_OVERLAP_MARKDOWN,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        exclude_headers: list[str] | None = None
    ):
        """Initialize the markdown splitter as per PRD specifications.

        Args:
            chunk_size: Target size for chunks (default: 1000 tokens as per PRD)
            chunk_overlap: Overlap for recursive splits (default: 200 tokens as per PRD)
            headers_to_split_on: Headers to split on (PRD default: H1-H4)
            exclude_headers: Headers to exclude (PRD default: Table of Contents, etc.)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # PRD requirement: configurable headers (default H1-H4)
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        # PRD requirement: noise filtering (default blocklist)
        self.exclude_headers = exclude_headers or [
            "Table of Contents",
            "Index",
            "Contributors"
        ]

    def split_text(self,
        text: str,
        metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Split markdown text into chunks using PRD two-pass strategy.

        Args:
            text: Markdown text to split
            metadata: Optional metadata to attach to chunks

        Returns:
            list of chunk dictionaries in PRD-specified format
        """
        # Store original lines for line number tracking
        original_lines = text.split('\n')

        # Pass 1: Split by headers (PRD requirement)
        sections = self._split_by_headers(text, original_lines)

        chunks = []
        for section in sections:
            section_chunks = self._split_large_section(section)

            for chunk_data in section_chunks:
                # PRD requirement: Create chunks with specified format
                chunk_id = create_chunk_id(
                    Path(metadata.get('source', 'unknown') if metadata else 'unknown'),
                    chunk_data.get('start_line', 1),
                    chunk_data['content']
                )

                chunk = {
                    "id": chunk_id,
                    "content": chunk_data['content'],
                    "token_count": estimate_token_count(chunk_data['content']),
                    "start_line": chunk_data.get('start_line', 1),
                    "end_line": chunk_data.get('end_line', 1),
                    "metadata": {
                        "filename": metadata.get('filename', '') if metadata else '',
                        "filepath": metadata.get('filepath', '') if metadata else '',
                        "source": metadata.get('source', '') if metadata else '',
                        **chunk_data.get('metadata', {})
                    },
                    "is_code": self._detect_code_content(chunk_data['content'])
                }
                chunks.append(chunk)

        return chunks

    def _split_by_headers(self, content: str, original_lines: list[str]) -> list[dict[str, Any]]:
        """Pass 1: Split by markdown headers using MarkdownHeaderTextSplitter

        Args:
            content: Full markdown content
            original_lines: Original file lines for line number tracking

        Returns:
            list of sections with line number information
        """

        # PRD requirement: Use MarkdownHeaderTextSplitter as the base
        # Note: MarkdownHeaderTextSplitter expects list of tuples format
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )

        splits = splitter.split_text(content)

        sections = []
        for split in splits:
            # Extract header metadata in PRD format (h1, h2, h3)
            metadata = {}
            for key, value in split.metadata.items():
                if key.startswith('Header '):
                    level = int(key.split(' ')[1])
                    metadata[f'h{level}'] = value

            # PRD requirement: Filter out noise (Table of Contents, etc.)
            header_values = [v for k, v in split.metadata.items() if k.startswith('Header ')]
            if header_values and any(header in self.exclude_headers for header in header_values):
                continue

            # Calculate line numbers by finding content in original file
            chunk_content = split.page_content
            start_line, end_line = self._find_line_numbers(chunk_content, original_lines)

            sections.append({
                'title': header_values[0] if header_values else 'Document',
                'level': max(1, len(header_values)),
                'content': chunk_content,
                'metadata': metadata,
                'start_line': start_line,
                'end_line': end_line
            })

        return sections

    def _find_line_numbers(
        self,
        chunk_content: str,
        original_lines: list[str]
    ) -> tuple[int, int]:
        """Find start and end line numbers for chunk content in original file.

        Args:
            chunk_content: The chunk text to locate
            original_lines: Original file lines

        Returns:
            Tuple of (start_line, end_line) with 1-based indexing
        """
        # Get first few significant words from chunk for matching
        chunk_lines = chunk_content.strip().split('\n')
        if not chunk_lines:
            return (1, 1)

        # Find the first non-empty line to search for
        first_line = None
        for line in chunk_lines:
            if line.strip():
                first_line = line.strip()
                break

        if not first_line:
            return (1, 1)

        # Search for this line in original file
        start_line = 1
        for i, orig_line in enumerate(original_lines, 1):
            if first_line in orig_line:
                start_line = i
                break

        # Estimate end line
        num_lines = len([line for line in chunk_lines if line.strip()])
        end_line = start_line + num_lines - 1

        return (start_line, max(start_line, end_line))

    def _build_header_metadata(
        self,
        header_stack: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build metadata dictionary in PRD format (h1, h2, h3)."""
        metadata = {}
        for header in header_stack:
            metadata[f'h{header["level"]}'] = header['title']
        return metadata

    def _split_large_section(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        """Pass 2: Apply RecursiveCharacterTextSplitter if section exceeds chunk_size"""
        content = section['content']

        # PRD requirement: If header section > Max_Chunk_Size, apply recursive splitting
        if estimate_token_count(content) <= self.chunk_size:
            return [section]

        # Check for code blocks
        if self._contains_code_blocks(content):
            logger.warning(f"Large section '{section['title']}' contains code blocks - keeping intact")
            return [section]

        return self._apply_recursive_split(section)


    def _apply_recursive_split(self, section: dict[str, Any]) -> list[dict[str, Any]]:
        """Apply RecursiveCharacterTextSplitter (PRD requirement)."""
        content = section['content']

        # PRD requirement: Apply RecursiveCharacterTextSplitter to chunks exceeding size
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=estimate_token_count
        )

        sub_chunks = splitter.split_text(content)

        # PRD requirement: All sub-chunks inherit parent header metadata
        chunks = []
        for sub_chunk in sub_chunks:
            chunk = {
                'content': sub_chunk,
                'metadata': section.get('metadata', {}).copy(),  # Inherit metadata
                'chunk_type': 'markdown_section_chunk'
            }
            chunks.append(chunk)

        return chunks


    def _contains_code_blocks(self, content: str) -> bool:
        """Check if content contains code blocks (PRD: never sever code blocks)."""
        return '```' in content

    def _detect_code_content(self, text: str) -> bool:
        """Detect if chunk content is primarily code."""
        code_indicators = [
            r'^\s*(def |class |import |from )',
            r'^\s*[{}();]',
            r'```\w*$',
            r'^\s*```',
        ]

        lines = text.split('\n')
        code_lines = sum(1 for line in lines
                        if any(re.search(pattern, line) for pattern in code_indicators))

        return code_lines / len(lines) > 0.3 if lines else False


def process_markdown_file(file_path: Path) -> list[dict[str, Any]]:
    """Process a markdown file using StructureAwareMarkdownSplitter (PRD compliant)."""
    try:
        # Use safe_read_file from utils
        content = safe_read_file(file_path)
        if not content or not content.strip():
            return []

        # Create splitter instance with PRD-configured parameters
        splitter = StructureAwareMarkdownSplitter(
            chunk_size=MAX_CHUNK_SIZE_MARKDOWN,
            chunk_overlap=CHUNK_OVERLAP_MARKDOWN,
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ],
            exclude_headers=[
                "Table of Contents",
                "Index",
                "Contributors",
                "Table of contents",
                "table of contents"
            ]
        )

        # Handle file path for production and testing
        try:
            relative_path = str(file_path.relative_to(HTTPX_REPO_DIR))
        except ValueError:
            relative_path = str(file_path)

        # Prepare metadata for the splitter
        metadata = {
            'source': relative_path,
            'filename': file_path.name
        }

        # Use the splitter to get chunks (PRD-compliant format)
        chunks = splitter.split_text(content, metadata)

        # Convert to unified schema format with Pydantic validation
        formatted_chunks = []
        for chunk in chunks:
            # Extract header metadata
            header_meta = {k: v for k, v in chunk['metadata'].items() if k.startswith('h')}

            try:
                # Create and validate chunk with Pydantic
                markdown_chunk = MarkdownChunk(
                    id=chunk['id'],
                    source_type="markdown",
                    chunk_type="markdown_section",
                    content=chunk['content'],
                    token_count=chunk['token_count'],
                    file_path=relative_path,
                    filename=file_path.name,
                    start_line=chunk.get('start_line', 1),
                    end_line=chunk.get('end_line', 1),
                    name=chunk['metadata'].get('h1', 'Document'),
                    full_name=chunk['metadata'].get('h1', 'Document'),
                    parent_context="",
                    headers=header_meta,
                    heading_level=max(1, len(header_meta)),
                    is_code=chunk['is_code']
                )
                # Convert to dict for JSON serialization
                formatted_chunks.append(markdown_chunk.model_dump(exclude_none=True))
            except Exception as e:
                logger.error(
                    f"Pydantic validation failed for markdown chunk "
                    f"in {file_path}:{chunk.get('start_line', 'unknown')}. "
                    f"Error: {e}. Skipping chunk - requires human examination."
                )
                # Skip invalid chunk

        return formatted_chunks

    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")
        return []


def main() -> int:
    """Main entry point.

    Returns:
        0 on success, 1 on failure
    """
    logger.info("Starting structure-aware markdown processing...")
    logger.info(f"Configuration: MAX_CHUNK_SIZE_MARKDOWN={MAX_CHUNK_SIZE_MARKDOWN}, CHUNK_OVERLAP_MARKDOWN={CHUNK_OVERLAP_MARKDOWN}")

    if not HTTPX_REPO_DIR.exists():
        logger.error(f"Repository not found at {HTTPX_REPO_DIR}")
        logger.error("Run clone_httpx_repo.py first")
        return 1

    # Find all markdown files
    markdown_files = list(HTTPX_REPO_DIR.rglob("*.md"))
    logger.info(f"Found {len(markdown_files)} Markdown files")

    # Filter excluded files using utils function
    included_files = [f for f in markdown_files if not should_exclude(f, HTTPX_REPO_DIR, REPO_EXCLUDE_PATTERNS)]
    logger.info(f"Processing {len(included_files)} files (excluded {len(markdown_files) - len(included_files)})")

    # Process files
    all_chunks = []
    total_input_tokens = 0

    for i, file_path in enumerate(included_files, 1):
        if i % 5 == 0 or i == 1:
            logger.info(f"Processing file {i}/{len(included_files)}: {file_path.name}")

        chunks = process_markdown_file(file_path)
        all_chunks.extend(chunks)

        # Track input tokens using safe_read_file
        content = safe_read_file(file_path)
        if content:
            total_input_tokens += estimate_token_count(content)

    logger.info(f"Created {len(all_chunks)} markdown chunks from {total_input_tokens} input tokens")

    # Save chunks to JSON using utils function
    if save_json_chunks(all_chunks, MARKDOWN_CHUNKS_FILE):
        logger.info(f"[SUCCESS] Saved chunks to {MARKDOWN_CHUNKS_FILE}")
    else:
        logger.error("[ERROR] Failed to save chunks")
        return 1

    # Generate and print statistics using utils functions
    stats = generate_processing_stats(all_chunks)
    print_processing_stats(stats, logger)

    return 0


if __name__ == "__main__":
    sys.exit(main())
