"""Router prompt for query decomposition.

This module contains the LLM prompt for the router component, which decomposes
complex user queries into structured retrieval requests.

Also provides codebase tree generation utility for router context.

Author: Hay Hoffman
Version: 1.3
"""

import logging
from pathlib import Path
from string import Template

from settings import REPO_EXCLUDE_PATTERNS, ROUTER_MAX_TREE_DEPTH

logger = logging.getLogger(__name__)

__all__ = ["ROUTER_PROMPT", "build_router_prompt", "generate_codebase_tree", "get_cached_codebase_tree"]

# Module-level cache for codebase tree (expensive to generate)
_codebase_tree_cache: dict[str, str] = {}


# Using Template for safe substitution (avoids issues with JSON braces)
ROUTER_PROMPT = Template("""You are a codebase navigation expert. Decompose user queries into
specific retrieval requests.

CODEBASE STRUCTURE:
$codebase_tree

USER QUERY: "$user_query"

TASK:
1. Clean the query from fluff/filler words
2. Identify ALL distinct pieces of information needed
3. For EACH piece, create a RetrievalRequest

CRITICAL RULES:
- Queries MUST be distinct (no rephrasing the same question)
- If queries target similar information → unify to single request with multiple filters
- Combine source_types in single request when possible
- Use folders/file_patterns only if query is specific
- Each request needs clear reasoning (for synthesis context)

FILTERING GUIDELINES:

1. Source Type Selection:
   - "code": Implementation, logic, algorithms, functions, classes
   - "markdown": Documentation, guides, explanations, README
   - "text": Configuration files (toml, yaml, json, ini)
   - Combine types when both implementation + docs needed

2. Folder Filtering (coarse):
   - Use when query mentions specific component/module
   - Example: "retrieval logic" → folders: ["src/retrieval/"]
   - Leave empty for broad/exploratory queries

3. File Pattern Filtering (fine-grained):
   - Use when query is very specific
   - Example: "README installation" → file_patterns: ["README.md"]
   - Use wildcards: ["*_search.py", "test_*.py"]
   - Leave empty for coarse search

EXAMPLES:

BAD (duplicate queries):
{
  "retrieval_requests": [
    {"query": "BM25 search implementation", "source_types": ["code"]},
    {"query": "BM25 documentation", "source_types": ["markdown"]}  // ❌ Redundant!
  ]
}

GOOD (unified):
{
  "retrieval_requests": [
    {
      "query": "BM25 search implementation and documentation",
      "source_types": ["code", "markdown"],  // ✅ Combined
      "folders": ["src/retrieval/", "documents/"],
      "file_patterns": [],
      "reasoning": "Need both implementation and documentation of BM25 algorithm"
    }
  ]
}

GOOD (multiple distinct queries - truly different topics):
User: "How does HTTPX implement sync vs async HTTP/2 support?"
{
  "retrieval_requests": [
    {
      "query": "HTTP/2 sync transport implementation HTTPTransport",
      "source_types": ["code"],
      "folders": ["httpx/_transports/"],
      "file_patterns": [],
      "reasoning": "Need synchronous HTTP/2 transport implementation details"
    },
    {
      "query": "AsyncHTTPTransport HTTP/2 async implementation",
      "source_types": ["code"],
      "folders": ["httpx/_transports/"],
      "file_patterns": [],
      "reasoning": "Need async HTTP/2 transport for comparison with sync version"
    }
  ]
}
// ✅ Valid: These are distinct code paths (sync vs async) requiring separate retrievals

OUTPUT FORMAT (JSON):
{
  "cleaned_query": "concise version of user query",
  "retrieval_requests": [
    {
      "query": "specific search query",
      "source_types": ["code", "markdown", "text"],
      "folders": ["path/to/folder/"],
      "file_patterns": ["*.md", "*_config.py"],
      "reasoning": "why this information is needed for answering the query"
    }
  ]
}""")


def build_router_prompt(user_query: str, codebase_tree: str) -> str:
    """Build router prompt with user query and codebase tree.

    Uses string.Template for safe substitution (avoids issues with JSON braces).

    Args:
        user_query: User's question/request
        codebase_tree: Directory tree representation of codebase

    Returns:
        Formatted prompt string ready for LLM

    Example:
        >>> tree = generate_codebase_tree(repo_root)
        >>> prompt = build_router_prompt("How does BM25 work?", tree)
    """
    return ROUTER_PROMPT.substitute(
        user_query=user_query,
        codebase_tree=codebase_tree,
    )


def generate_codebase_tree(
    repo_root: Path,
    max_depth: int = ROUTER_MAX_TREE_DEPTH,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Generate compact directory tree for router context.

    Produces a filtered directory tree representation that helps the LLM
    router understand codebase structure for intelligent query decomposition.

    Args:
        repo_root: Repository root path
        max_depth: Maximum depth of tree (default: 4 from ROUTER_MAX_TREE_DEPTH)
        exclude_patterns: Patterns to exclude (default: from REPO_EXCLUDE_PATTERNS)

    Returns:
        Compact directory tree string (~500-800 tokens)

    Example:
        >>> tree = generate_codebase_tree(Path("/my/repo"))
        >>> print(tree)
        my_repo/
        ├── src/
        │   ├── retrieval/
        │   │   ├── hybrid_retriever.py
        │   │   └── faiss_store.py
        │   └── agent/
        │       └── nodes/
        ├── prompts/
        │   ├── router_prompt.py
        │   └── synthesis_prompt.py
        └── tests/
    """
    if exclude_patterns is None:
        exclude_patterns = REPO_EXCLUDE_PATTERNS

    def _should_exclude(path: Path) -> bool:
        """Check if path matches exclude patterns."""
        for pattern in exclude_patterns:
            if path.match(pattern):
                return True
        return False

    def _build_tree(directory: Path, prefix: str, depth: int) -> list[str]:
        """Recursively build tree structure."""
        if depth > max_depth:
            return []

        lines = []
        try:
            entries = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError:
            return []

        # Filter out excluded paths
        entries = [e for e in entries if not _should_exclude(e)]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                lines.extend(_build_tree(entry, prefix + extension, depth + 1))
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

        return lines

    # Build tree
    tree_lines = [f"{repo_root.name}/"]
    tree_lines.extend(_build_tree(repo_root, "", 0))

    return "\n".join(tree_lines)


def get_cached_codebase_tree(repo_root: Path) -> str:
    """Get cached codebase tree (generates once per repo_root).

    This function caches the expensive tree generation operation.
    Use this instead of generate_codebase_tree for repeated calls.

    Args:
        repo_root: Repository root path

    Returns:
        Cached codebase tree string

    Example:
        >>> tree = get_cached_codebase_tree(Path("/my/repo"))
        >>> # Second call returns cached result
        >>> tree2 = get_cached_codebase_tree(Path("/my/repo"))
    """
    repo_key = str(repo_root)
    if repo_key not in _codebase_tree_cache:
        logger.info(f"Generating codebase tree for: {repo_root}")
        _codebase_tree_cache[repo_key] = generate_codebase_tree(repo_root)

    return _codebase_tree_cache[repo_key]
