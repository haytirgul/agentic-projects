"""
Web search query generator using LLM.

Transforms user queries and intent into optimized search queries for web search APIs.
"""

from typing import List
import logging

from models.intent_classification import IntentClassification
from models.rag_models import WebSearchQueries
from prompts.web_search_query_generation import build_web_search_query_messages
from settings import LEVEL_TO_MODEL
from src.llm import get_cached_llm

logger = logging.getLogger(__name__)

__all__ = ["WebSearchQueryGenerator"]


class WebSearchQueryGenerator:
    """Generate optimized web search queries from user intent."""

    def __init__(self, llm=None):
        """
        Initialize query generator.

        Args:
            llm: LLM instance (defaults to cached fast model)
        """
        self.llm = llm or get_cached_llm(LEVEL_TO_MODEL["fast"])

    def generate_queries(
        self,
        user_query: str,
        intent: IntentClassification
    ) -> List[str]:
        """
        Generate 1-3 optimized search queries using structured output.

        Args:
            user_query: Raw user query
            intent: Intent classification result

        Returns:
            List of optimized search queries (1-3 queries)

        Example:
            >>> generator = WebSearchQueryGenerator()
            >>> intent = IntentClassification(
            ...     framework="langgraph",
            ...     language="python",
            ...     keywords=["checkpointer"],
            ...     topics=["persistence"],
            ...     intent_type="factual_lookup",
            ...     requires_rag=True
            ... )
            >>> queries = generator.generate_queries("What is a checkpointer?", intent)
            >>> len(queries)
            2
            >>> "LangGraph" in queries[0]
            True
        """
        try:
            # Build structured messages
            messages = build_web_search_query_messages(
                user_query=user_query,
                intent=intent
            )

            # Create LLM with structured output
            structured_llm = self.llm.with_structured_output(WebSearchQueries)

            # Get structured response
            result: WebSearchQueries = structured_llm.invoke(messages)

            # Validate result
            if not result.queries:
                logger.warning("LLM returned empty queries, falling back to basic query")
                queries = [self._generate_fallback_query(user_query, intent)]
            else:
                queries = result.queries

            logger.info(f"Generated {len(queries)} search queries")
            logger.debug(f"Queries: {queries}")
            logger.debug(f"Reasoning: {result.reasoning}")

            return queries[:3]  # Max 3 queries

        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Fallback to basic query
            return [self._generate_fallback_query(user_query, intent)]

    def _generate_fallback_query(
        self,
        user_query: str,
        intent: IntentClassification
    ) -> str:
        """
        Generate basic search query without LLM.

        Args:
            user_query: Raw user query
            intent: Intent classification result

        Returns:
            Basic search query string
        """
        parts = []

        # Add framework
        if intent.framework and intent.framework != "general":
            parts.append(intent.framework.capitalize())

        # Add language
        if intent.language:
            parts.append(intent.language.capitalize())

        # Add user query
        parts.append(user_query)

        fallback_query = " ".join(parts)
        logger.info(f"Using fallback query: {fallback_query}")

        return fallback_query
