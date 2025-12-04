"""Process httpx repository into code chunks using AST-based chunking.

This script walks through the httpx repository, parses Python files using AST,
and creates semantic code chunks (functions, classes, modules).

Usage:
    python scripts/data_pipeline/processing/chunk_code.py
"""

import ast
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from settings import HTTPX_REPO_DIR, DATA_DIR, REPO_EXCLUDE_PATTERNS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

OUTPUT_FILE = DATA_DIR / "processed" / "code_chunks.json"


def should_exclude(file_path: Path, repo_root: Path) -> bool:
    """Check if file should be excluded based on patterns.

    Args:
        file_path: Path to the file
        repo_root: Root of the repository

    Returns:
        True if file should be excluded
    """
    try:
        relative_path = file_path.relative_to(repo_root)
        relative_str = str(relative_path)

        for pattern in REPO_EXCLUDE_PATTERNS:
            # Simple pattern matching (not full glob)
            if pattern.startswith("*"):
                if relative_str.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("/*"):
                if relative_str.startswith(pattern[:-2]):
                    return True
            elif pattern in relative_str:
                return True

        return False
    except ValueError:
        return True


def extract_docstring(node: ast.AST) -> str:
    """Extract docstring from AST node.

    Args:
        node: AST node (FunctionDef, ClassDef, or Module)

    Returns:
        Docstring text or empty string
    """
    docstring = ast.get_docstring(node)
    return docstring if docstring else ""


def chunk_function(func_node: ast.FunctionDef, file_path: Path, source_lines: List[str],
                   class_name: str = None) -> Dict[str, Any]:
    """Create a chunk from a function definition.

    Args:
        func_node: AST FunctionDef node
        file_path: Path to the source file
        source_lines: Lines of source code
        class_name: Name of parent class if function is a method

    Returns:
        Chunk dictionary
    """
    start_line = func_node.lineno
    end_line = func_node.end_lineno

    # Get function code
    code_lines = source_lines[start_line - 1:end_line]
    code_content = "".join(code_lines)

    # Build context
    context = f"{class_name}.{func_node.name}" if class_name else func_node.name

    # Extract imports from top of file
    imports = []
    for line in source_lines[:20]:  # Check first 20 lines for imports
        if line.strip().startswith(("import ", "from ")):
            imports.append(line.strip())

    return {
        "file_path": str(file_path.relative_to(HTTPX_REPO_DIR)),
        "start_line": start_line,
        "end_line": end_line,
        "chunk_type": "function",
        "name": func_node.name,
        "parent_context": class_name or "",
        "full_name": context,
        "content": code_content,
        "docstring": extract_docstring(func_node),
        "imports": imports[:5],  # Limit to first 5 imports
    }


def chunk_class(class_node: ast.ClassDef, file_path: Path, source_lines: List[str]) -> List[Dict[str, Any]]:
    """Create chunks from a class definition.

    Creates one chunk for the class, plus chunks for each method.

    Args:
        class_node: AST ClassDef node
        file_path: Path to the source file
        source_lines: Lines of source code

    Returns:
        List of chunk dictionaries
    """
    chunks = []
    start_line = class_node.lineno
    end_line = class_node.end_lineno

    # Get class code
    code_lines = source_lines[start_line - 1:end_line]
    code_content = "".join(code_lines)

    # Extract imports
    imports = []
    for line in source_lines[:20]:
        if line.strip().startswith(("import ", "from ")):
            imports.append(line.strip())

    # Class-level chunk
    chunks.append({
        "file_path": str(file_path.relative_to(HTTPX_REPO_DIR)),
        "start_line": start_line,
        "end_line": end_line,
        "chunk_type": "class",
        "name": class_node.name,
        "parent_context": "",
        "full_name": class_node.name,
        "content": code_content,
        "docstring": extract_docstring(class_node),
        "imports": imports[:5],
    })

    # Method chunks
    for node in ast.walk(class_node):
        if isinstance(node, ast.FunctionDef) and node.lineno > class_node.lineno:
            method_chunk = chunk_function(node, file_path, source_lines, class_node.name)
            chunks.append(method_chunk)

    return chunks


def process_file(file_path: Path) -> List[Dict[str, Any]]:
    """Process a Python file into code chunks.

    Args:
        file_path: Path to Python file

    Returns:
        List of chunk dictionaries
    """
    chunks = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            source_lines = source_code.splitlines(keepends=True)

        # Parse AST
        tree = ast.parse(source_code, filename=str(file_path))

        # Extract top-level classes and functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                chunks.extend(chunk_class(node, file_path, source_lines))
            elif isinstance(node, ast.FunctionDef):
                chunks.append(chunk_function(node, file_path, source_lines))

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
    included_files = [f for f in python_files if not should_exclude(f, HTTPX_REPO_DIR)]
    logger.info(f"Processing {len(included_files)} files (excluded {len(python_files) - len(included_files)})")

    # Process files
    all_chunks = []
    for i, file_path in enumerate(included_files, 1):
        if i % 10 == 0:
            logger.info(f"Processing file {i}/{len(included_files)}: {file_path.name}")

        chunks = process_file(file_path)
        all_chunks.extend(chunks)

    logger.info(f"Created {len(all_chunks)} code chunks")

    # Save chunks to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved chunks to {OUTPUT_FILE}")
    logger.info(f"Total chunks: {len(all_chunks)}")

    # Statistics
    chunk_types = {}
    for chunk in all_chunks:
        chunk_type = chunk['chunk_type']
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

    logger.info("Chunk type distribution:")
    for chunk_type, count in sorted(chunk_types.items()):
        logger.info(f"  {chunk_type}: {count}")


if __name__ == "__main__":
    main()
