"""Router prompt for query decomposition.

This module contains the LLM prompt for the router component, which decomposes
complex user queries into structured retrieval requests.

Also provides codebase tree generation utility for router context.

Author: Hay Hoffman
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
ROUTER_PROMPT = Template("""### PERSONA
You are a Codebase Navigation Expert that decomposes user queries into precise retrieval requests.

### TASK
Analyze the user query and codebase structure to create 1-4 targeted retrieval requests.

CODEBASE STRUCTURE:
$codebase_tree

USER QUERY: "$user_query"

### CRITICAL RULES
1. **Semantic Distinctness**: Each query MUST target a fundamentally different concept
   - SAME concept = combine into ONE request (even if different source types)
   - DIFFERENT concepts = separate requests
2. **Request Count**: Generate 1-4 requests maximum
   - Simple/focused queries → 1 request
   - Multi-concept queries → 2-4 requests (one per distinct concept)
3. **Folder Inference**: ALWAYS infer folders from the codebase tree above
   - Never leave folders empty unless query is truly codebase-wide
4. **Source Type Combination**: Combine source_types in a single request when targeting the same concept

### SOURCE TYPES
- "code": Implementation, functions, classes, algorithms
- "markdown": Documentation, guides, README files
- "text": Configuration files (toml, yaml, json, ini)

### FOLDER INFERENCE
- Match query topic to folder names in codebase tree
- Use parent folders if unsure (broader is safer than wrong)
- Leave empty ONLY for codebase-wide queries (e.g., "list all files")

### FILE PATTERNS (optional)
- Use when query mentions specific files: ["README.md", "test_*.py"]
- Use wildcards for patterns: ["*_client.py"]
- Leave empty for folder-level filtering (default)

### EDGE CASES
- **Simple query** (single concept): Return exactly 1 request
- **Off-topic query** (not about codebase): Return 1 request with broad folders, let retrieval handle it
- **Ambiguous query**: Interpret most likely intent, use broader folders
- **Conversational fluff** ("Can you help me understand..."): Strip to core question

### EXAMPLES

EXAMPLE 1 - Single Concept (combine source types):
User Query: "How does BM25 search work?"
{
  "cleaned_query": "BM25 search algorithm",
  "retrieval_requests": [
    {
      "query": "BM25 search implementation",
      "source_types": ["code", "markdown"],
      "folders": ["src/retrieval/", "documents/"],
      "file_patterns": [],
      "reasoning": "BM25 is a single algorithm - need both implementation and docs"
    }
  ]
}

EXAMPLE 2 - Multiple Distinct Concepts:
User Query: "How do timeouts and caching work in the HTTP client?"
{
  "cleaned_query": "HTTP client timeout and caching mechanisms",
  "retrieval_requests": [
    {
      "query": "HTTP client timeout configuration and handling",
      "source_types": ["code"],
      "folders": ["src/client/", "src/transport/"],
      "file_patterns": [],
      "reasoning": "Timeout = request abortion mechanism"
    },
    {
      "query": "HTTP response caching implementation",
      "source_types": ["code"],
      "folders": ["src/cache/", "src/client/"],
      "file_patterns": [],
      "reasoning": "Caching = response storage/retrieval - different subsystem"
    }
  ]
}

EXAMPLE 3 - Simple Direct Query:
User Query: "What does the Config class do?"
{
  "cleaned_query": "Config class purpose",
  "retrieval_requests": [
    {
      "query": "Config class definition and usage",
      "source_types": ["code"],
      "folders": ["src/", "config/"],
      "file_patterns": ["*config*.py"],
      "reasoning": "Single class lookup - find definition and docstring"
    }
  ]
}

EXAMPLE 4 - Documentation Query:
User Query: "How do I set up the project?"
{
  "cleaned_query": "project setup instructions",
  "retrieval_requests": [
    {
      "query": "project setup installation guide",
      "source_types": ["markdown"],
      "folders": ["documents/", ""],
      "file_patterns": ["README.md", "SETUP.md", "*setup*"],
      "reasoning": "Setup instructions are typically in markdown docs"
    }
  ]
}

### ANTI-PATTERNS (DO NOT DO)
❌ Multiple queries for same concept with different source types
❌ More than 4 retrieval requests
❌ Empty folders array for specific queries
❌ Queries that are just rewordings of each other

### OUTPUT SCHEMA
{
  "cleaned_query": "string - core intent without filler words",
  "retrieval_requests": [
    {
      "query": "string - specific search query",
      "source_types": ["code|markdown|text"],
      "folders": ["array of folder paths from codebase tree"],
      "file_patterns": ["optional file patterns with wildcards"],
      "reasoning": "string - why this information is needed"
    }
  ]
}

### VALIDATION CHECKLIST
Before output, verify:
☐ 1-4 requests only
☐ Each request targets a DIFFERENT concept
☐ Folders are inferred from codebase tree (not empty unless codebase-wide)
☐ Source types combined when same concept
☐ cleaned_query captures core intent""")


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
