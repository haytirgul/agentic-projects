"""
Deduplication logic for web search results.

Removes web results that duplicate offline documentation to avoid redundancy.
"""

from typing import List
import logging
from src.rag.web_search_api import WebSearchResult
from models.rag_models import RetrievedDocument

logger = logging.getLogger(__name__)

__all__ = ["deduplicate_web_results", "OFFLINE_DOC_DOMAINS"]


# Known documentation domains we have offline
OFFLINE_DOC_DOMAINS = [
    "python.langchain.com",
    "langchain.com",
    "langchain-ai.github.io",
    "docs.langchain.com",
    "js.langchain.com",
    "api.python.langchain.com",
    "api.js.langchain.com",
]


def deduplicate_web_results(
    web_results: List[WebSearchResult],
    offline_docs: List[RetrievedDocument]
) -> List[WebSearchResult]:
    """
    Remove web results that duplicate offline documentation.

    Strategy:
    1. Remove results from known offline domains
    2. Remove results with titles matching offline doc titles (fuzzy match)

    Args:
        web_results: Web search results
        offline_docs: Retrieved offline documents

    Returns:
        Filtered web results

    Example:
        >>> web = [WebSearchResult(
        ...     title="LangChain Docs",
        ...     url="https://python.langchain.com/foo",
        ...     content="...",
        ...     score=0.9
        ... )]
        >>> offline = [RetrievedDocument(title="LangChain Docs", ...)]
        >>> filtered = deduplicate_web_results(web, offline)
        >>> len(filtered)
        0  # Removed because domain is in OFFLINE_DOC_DOMAINS
    """
    if not web_results:
        return []

    filtered = []

    # Build set of offline doc titles (lowercase for fuzzy matching)
    offline_titles = {doc.title.lower() for doc in offline_docs}

    for result in web_results:
        # Check domain - ensure it's actually a domain match, not just substring
        url_lower = result.url.lower()
        is_offline_domain = any(
            # Check if URL starts with domain or contains domain as subdomain
            url_lower.startswith(f"https://{domain}/") or
            url_lower.startswith(f"http://{domain}/") or
            f".{domain}/" in url_lower
            for domain in OFFLINE_DOC_DOMAINS
        )

        if is_offline_domain:
            logger.debug(f"Skipping web result (offline domain): {result.url}")
            continue

        # Check title similarity
        title_lower = result.title.lower()
        is_duplicate_title = any(
            _fuzzy_title_match(title_lower, offline_title)
            for offline_title in offline_titles
        )

        if is_duplicate_title:
            logger.debug(f"Skipping web result (duplicate title): {result.title}")
            continue

        filtered.append(result)

    dedup_count = len(web_results) - len(filtered)
    if dedup_count > 0:
        logger.info(f"Deduplicated: {len(web_results)} → {len(filtered)} web results ({dedup_count} removed)")
    else:
        logger.info(f"No duplicates found, keeping all {len(web_results)} web results")

    return filtered


def _fuzzy_title_match(title1: str, title2: str, threshold: float = 0.8) -> bool:
    """
    Check if two titles are similar using fuzzy matching.

    Args:
        title1: First title (lowercase)
        title2: Second title (lowercase)
        threshold: Similarity threshold (0.0-1.0)

    Returns:
        True if titles are similar above threshold

    Example:
        >>> _fuzzy_title_match("langchain python tutorial", "langchain python guide")
        True
        >>> _fuzzy_title_match("langchain", "langgraph")
        False
    """
    try:
        from rapidfuzz import fuzz
        similarity = fuzz.ratio(title1, title2) / 100.0  # Convert to 0-1 range
        return similarity >= threshold
    except ImportError:
        # Fallback to exact match if rapidfuzz not available
        logger.warning("rapidfuzz not available, using exact title matching")
        return title1 == title2
