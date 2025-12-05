"""Process httpx repository into code chunks using AST-based chunking.

This script walks through the httpx repository, parses Python files using AST,
and creates semantic code chunks (functions, classes, modules).

Usage:
    python scripts/data_pipeline/processing/chunk_code.py
"""

import ast
import logging
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

__all__: list[str] = []

# Setup project paths first - must be done before importing from settings
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    create_chunk_id,
    estimate_token_count,
    generate_processing_stats,
    print_processing_stats,
    save_json_chunks,
    setup_project_paths,
    should_exclude,
)

# Initialize project and logging
project_root = setup_project_paths()
logger = logging.getLogger(__name__)

from models.chunk import CodeChunk
from settings import CODE_CHUNKS_FILE, HTTPX_REPO_DIR, REPO_EXCLUDE_PATTERNS

# Constants
MAX_TOKENS = 1000  # Token limit for chunks


def get_signature(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef, source_lines: list[str]
) -> str:
    """Extract the function signature (definition line only).

    Args:
        func_node: AST FunctionDef or AsyncFunctionDef node
        source_lines: Lines of source code

    Returns:
        Function signature string (e.g., "def handle_response(self,
        response: Response) -> None:")
    """
    line_no = func_node.lineno - 1  # Convert to 0-based indexing
    return source_lines[line_no].rstrip()


def get_file_imports(tree: ast.Module) -> list[str]:
    """Extract all top-level import statements from AST tree.

    Args:
        tree: AST module tree

    Returns:
        List of import statement strings
    """
    imports = []

    # Only get imports from module body (top-level)
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(
                f"import {', '.join(alias.name for alias in node.names)}"
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = ', '.join(alias.name for alias in node.names)
            imports.append(f"from {module} import {names}")

    return imports[:10]  # Limit to first 10 imports


def split_large_chunk(chunk: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a large chunk into smaller pieces using RecursiveCharacterTextSplitter.

    This implements the PRD requirement to use RecursiveCharacterTextSplitter
    for oversized chunks while preserving all metadata.

    Args:
        chunk: Original chunk dictionary

    Returns:
        List of smaller chunk dictionaries with preserved metadata
    """
    content = chunk["content"]

    # PRD requirement: Use RecursiveCharacterTextSplitter
    # Token-based splitting (MAX_TOKENS = 1000)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_TOKENS * 4,  # Approximate characters for 1000 tokens
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split the content
    sub_contents = splitter.split_text(content)

    # If only one chunk, return original
    if len(sub_contents) == 1:
        return [chunk]

    # Create sub-chunks with preserved metadata (PRD requirement)
    sub_chunks = []
    for i, sub_content in enumerate(sub_contents, 1):
        sub_chunk = chunk.copy()
        sub_chunk["content"] = sub_content
        sub_chunk["chunk_id"] = f"{chunk.get('chunk_id', chunk['full_name'])}_part_{i}"
        # PRD requirement: All sub-chunks must retain parent metadata
        sub_chunk["parent_chunk"] = chunk["full_name"]
        sub_chunks.append(sub_chunk)

    return sub_chunks





def extract_docstring(
    node: ast.FunctionDef | ast.ClassDef | ast.Module | ast.AsyncFunctionDef,
) -> str:
    """Extract docstring from AST node.

    Args:
        node: AST node (FunctionDef, ClassDef, or Module)

    Returns:
        Docstring text or empty string
    """
    docstring = ast.get_docstring(node)
    return docstring if docstring else ""


def chunk_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: Path,
    source_lines: list[str],
    class_name: str | None = None,
    tree: ast.Module | None = None,
) -> dict[str, Any]:
    """Create a chunk from a function definition.

    Args:
        func_node: AST FunctionDef or AsyncFunctionDef node
        file_path: Path to the source file
        source_lines: Lines of source code
        class_name: Name of parent class if function is a method
        tree: AST module tree for import extraction

    Returns:
        Chunk dictionary compatible with CodeChunk model
    """
    start_line = func_node.lineno
    end_line = func_node.end_lineno

    # Get function code
    code_lines = source_lines[start_line - 1:end_line]
    code_content = "".join(code_lines)

    # Build context
    context = f"{class_name}.{func_node.name}" if class_name else func_node.name

    # Extract imports using AST if tree is provided
    imports = get_file_imports(tree) if tree else []

    # Handle file path (for testing and production)
    try:
        relative_path = str(file_path.relative_to(HTTPX_REPO_DIR))
    except ValueError:
        # For test files or files outside HTTPX_REPO_DIR
        relative_path = str(file_path)

    # Generate unique chunk ID
    chunk_id = create_chunk_id(file_path, start_line, code_content)

    # Calculate token count
    token_count = estimate_token_count(code_content)

    return {
        "id": chunk_id,
        "source_type": "code",
        "chunk_type": "function",
        "content": code_content,
        "token_count": token_count,
        "file_path": relative_path,
        "filename": file_path.name,
        "start_line": start_line,
        "end_line": end_line,
        "name": func_node.name,
        "full_name": context,
        "parent_context": class_name or "",
        "docstring": extract_docstring(func_node),
        "imports": imports[:5],  # Limit to first 5 imports
    }


def chunk_class(
    class_node: ast.ClassDef,
    file_path: Path,
    source_lines: list[str],
    tree: ast.Module,
) -> list[dict[str, Any]]:
    """Create chunks from a class definition.

    Creates one chunk for the class (signature-only summary), plus chunks for each method.

    Args:
        class_node: AST ClassDef node
        file_path: Path to the source file
        source_lines: Lines of source code
        tree: AST module tree for import extraction

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    start_line = class_node.lineno
    end_line = class_node.end_lineno

    # Extract imports using AST
    imports = get_file_imports(tree)

    # Build signature-only class summary
    class_content_parts = []

    # 1. Class definition line
    class_def_line = source_lines[start_line - 1].rstrip()
    class_content_parts.append(class_def_line)

    # 2. Class docstring
    docstring = extract_docstring(class_node)
    if docstring:
        # Add docstring with proper indentation
        docstring_lines = docstring.split('\n')
        class_content_parts.append(f'    """{docstring_lines[0]}')
        for line in docstring_lines[1:]:
            class_content_parts.append(f'    {line}')
        class_content_parts.append('    """')

    # 3. Method signatures
    for node in ast.walk(class_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno > class_node.lineno:
            # Check if this is a direct child (not nested in another function/class)
            is_direct_method = True
            for child in ast.iter_child_nodes(class_node):
                if isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and child != node:
                    if child.end_lineno and child.lineno < node.lineno < child.end_lineno:
                        is_direct_method = False
                        break
            if is_direct_method:
                signature = get_signature(node, source_lines)
                class_content_parts.append(f"    {signature}")

    # Join all parts
    class_content = "\n".join(class_content_parts)

    # Class-level chunk (signature-only)
    try:
        relative_path = str(file_path.relative_to(HTTPX_REPO_DIR))
    except ValueError:
        # For test files or files outside HTTPX_REPO_DIR
        relative_path = str(file_path)

    # Generate unique chunk ID
    chunk_id = create_chunk_id(file_path, start_line, class_content)

    # Calculate token count
    token_count = estimate_token_count(class_content)

    class_chunk = {
        "id": chunk_id,
        "source_type": "code",
        "chunk_type": "class",
        "content": class_content,
        "token_count": token_count,
        "file_path": relative_path,
        "filename": file_path.name,
        "start_line": start_line,
        "end_line": end_line,
        "name": class_node.name,
        "full_name": class_node.name,
        "parent_context": "",
        "docstring": docstring,
        "imports": imports[:5],
    }

    # Check if class chunk needs splitting
    if len(class_content) > MAX_TOKENS * 4:  # Rough token approximation
        chunks.extend(split_large_chunk(class_chunk))
    else:
        chunks.append(class_chunk)

    # Method chunks (full content)
    for node in ast.walk(class_node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno > class_node.lineno:
            method_chunk = chunk_function(node, file_path, source_lines, class_node.name, tree)
            # Check if method chunk needs splitting
            if len(method_chunk["content"]) > MAX_TOKENS * 4:  # Rough token approximation
                chunks.extend(split_large_chunk(method_chunk))
            else:
                chunks.append(method_chunk)

    return chunks


def process_file(file_path: Path) -> list[dict[str, Any]]:
    """Process a Python file into code chunks.

    Args:
        file_path: Path to Python file

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    try:
        with open(file_path, encoding='utf-8') as f:
            source_code = f.read()
            source_lines = source_code.splitlines(keepends=True)

        # Parse AST
        tree = ast.parse(source_code, filename=str(file_path))

        # Extract top-level classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.extend(chunk_class(node, file_path, source_lines, tree))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(chunk_function(node, file_path, source_lines, tree=tree))

    except SyntaxError as e:
        logger.warning(f"Syntax error in {file_path}: {e}")
    except Exception as e:
        logger.warning(f"Error processing {file_path}: {e}")

    return chunks


def main():
    """Main entry point."""
    logger.info("Starting code chunking process...")

    if not HTTPX_REPO_DIR.exists():
        logger.error(f"Repository not found at {HTTPX_REPO_DIR}")
        logger.error("Run clone_httpx_repo.py first")
        sys.exit(1)

    # Find all Python files
    python_files = list(HTTPX_REPO_DIR.rglob("*.py"))
    logger.info(f"Found {len(python_files)} Python files")

    # Filter excluded files
    included_files = [f for f in python_files if not should_exclude(f, HTTPX_REPO_DIR, REPO_EXCLUDE_PATTERNS)]
    logger.info(f"Processing {len(included_files)} files (excluded {len(python_files) - len(included_files)})")

    # Process files
    all_chunks = []
    for i, file_path in enumerate(included_files, 1):
        if i % 10 == 0:
            logger.info(f"Processing file {i}/{len(included_files)}: {file_path.name}")

        chunks = process_file(file_path)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} code chunks")

    # Save chunks to JSON using utility function
    if save_json_chunks(all_chunks, CODE_CHUNKS_FILE):
        logger.info(f"✅ Saved chunks to {CODE_CHUNKS_FILE}")
    else:
        logger.error("❌ Failed to save chunks")
        return

    # Generate and print statistics using utility functions
    stats = generate_processing_stats(all_chunks)
    print_processing_stats(stats, logger)


if __name__ == "__main__":
    main()
