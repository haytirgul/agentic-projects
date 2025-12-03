"""
Web search API integration using Tavily.

Provides clean, parsed web search results optimized for RAG applications.
"""

from typing import List, Optional
import logging
import asyncio

from settings import AGENT_MODE

logger = logging.getLogger(__name__)

__all__ = ["WebSearchResult", "WebSearchEngine"]


class WebSearchResult:
    """A single web search result."""

    def __init__(
        self,
        title: str,
        url: str,
        content: str,
        raw_content: Optional[str] = None,
        score: float = 0.0
    ):
        """
        Initialize web search result.

        Args:
            title: Page title
            url: Page URL
            content: Clean extracted content
            raw_content: Full HTML (optional)
            score: Relevance score from search engine
        """
        self.title = title
        self.url = url
        self.content = content
        self.raw_content = raw_content
        self.score = score

    def __repr__(self) -> str:
        return f"WebSearchResult(title='{self.title}', url='{self.url}', score={self.score:.2f}, content='{self.content}')"


class WebSearchEngine:
    """Web search using Tavily API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search engine.

        Only initializes Tavily client if AGENT_MODE is 'online'.

        Args:
            api_key: Tavily API key

        Raises:
            ValueError: If API key is missing and AGENT_MODE is online
            ImportError: If tavily-python is not installed and AGENT_MODE is online
        """
        # Only initialize if in online mode
        if AGENT_MODE != "online":
            logger.info(f"Web search disabled: AGENT_MODE='{AGENT_MODE}' (must be 'online')")
            self.client = None
            return

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is required for web search when AGENT_MODE='online'. "
                "Get one at https://tavily.com (free tier: 1000 searches/month)"
            )

        try:
            from tavily import TavilyClient
            self.client = TavilyClient(api_key=api_key)
            logger.info("Tavily client initialized successfully for online mode")
        except ImportError:
            raise ImportError(
                "tavily-python is required for web search when AGENT_MODE='online'. "
                "Install it with: pip install tavily-python"
            )

    async def search(
        self,
        queries: List[str],
        max_results_per_query: int = 5
    ) -> List[WebSearchResult]:
        """
        Execute web search for multiple queries in parallel.

        Args:
            queries: List of search queries
            max_results_per_query: Max results per query

        Returns:
            List of WebSearchResult objects (deduplicated by URL)

        Raises:
            RuntimeError: If web search is called when AGENT_MODE is not 'online'

        Example:
            >>> engine = WebSearchEngine(api_key="your_key")
            >>> queries = ["LangGraph Python checkpointer tutorial"]
            >>> results = await engine.search(queries, max_results_per_query=3)
            >>> len(results) <= 3
            True
            >>> all(r.url.startswith("http") for r in results)
            True
        """
        if self.client is None:
            raise RuntimeError(
                f"Web search is not available: AGENT_MODE='{AGENT_MODE}' (must be 'online'). "
                "Set AGENT_MODE=online and provide TAVILY_API_KEY to enable web search."
            )

        async def _search_single_query(query: str) -> List[WebSearchResult]:
            """Search for a single query and return results."""
            try:
                logger.info(f"Searching web: {query}")

                # Tavily only has synchronous search, so run it in executor
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=max_results_per_query,
                        include_raw_content=True
                    )
                )

                results = []
                for result in response.get("results", []):
                    url = result.get("url", "")
                    results.append(WebSearchResult(
                        title=result.get("title", "Untitled"),
                        url=url,
                        content=result.get("content", ""),  # Tavily pre-extracts content
                        score=result.get("score", 0.0)
                    ))
                return results
            except Exception as e:
                logger.error(f"Web search failed for query '{query}': {e}")
                return []

        # Execute all queries in parallel
        logger.debug(f"Executing {len(queries)} web searches in parallel...")
        query_tasks = [asyncio.create_task(_search_single_query(query)) for query in queries]
        query_results = await asyncio.gather(*query_tasks, return_exceptions=True)

        # Collect and deduplicate results
        all_results = []
        seen_urls = set()

        for query_result in query_results:
            if isinstance(query_result, Exception):
                logger.error(f"Query task failed with exception: {query_result}")
                continue

            for result in query_result:
                if result.url in seen_urls:
                    logger.debug(f"Skipping duplicate URL: {result.url}")
                    continue

                seen_urls.add(result.url)
                all_results.append(result)

        logger.info(f"Found {len(all_results)} unique web results across {len(queries)} queries")
        return all_results

    async def search_single(self, query: str, max_results: int = 5) -> List[WebSearchResult]:
        """
        Search with a single query (convenience method).

        Args:
            query: Search query
            max_results: Max results to return

        Returns:
            List of WebSearchResult objects

        Raises:
            RuntimeError: If web search is called when AGENT_MODE is not 'online'
        """
        return await self.search([query], max_results_per_query=max_results)
