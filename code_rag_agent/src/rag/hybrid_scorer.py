"""Simplified hybrid scoring for RAG retrieval.

This module provides a clean, maintainable implementation of hybrid search scoring
that combines vector, BM25, and web search results.

Key improvements over previous implementation:
- Single responsibility: scoring logic in one place
- Consistent data structures: all results as RetrievedDocument objects
- Clear scoring formula with documented weights
- Easy to test and tune
"""

from typing import List, Any, Dict
import logging
from models.rag_models import RetrievedDocument
from models.intent_classification import IntentClassification
from settings import VECTOR_WEIGHT, BM25_WEIGHT, WEB_BONUS

logger = logging.getLogger(__name__)

__all__ = ["HybridScorer"]


# Reranking configuration (fast, real-time methods only)
ENABLE_RRF_RERANKING = True  # Reciprocal Rank Fusion reranking
RRF_K_PARAMETER = 60  # Standard RRF constant (controls rank position influence)


class HybridScorer:
    """
    Combines vector, BM25, and web search results using weighted scoring + RRF reranking.

    This class provides fast, real-time hybrid search with two scoring approaches:
    1. **Weighted Linear Combination** (default): Vector * weight + BM25 * weight + Web * weight
    2. **Reciprocal Rank Fusion (RRF)** (optional): Rank-based fusion algorithm

    RRF is parameter-free and often outperforms weighted combinations without
    needing score normalization. It's fast (O(n)) and works well when individual
    retriever scores are unreliable.

    Example:
        >>> scorer = HybridScorer()
        >>> docs = scorer.score_and_rank(
        ...     vector_results=vector_matches,
        ...     bm25_results=bm25_matches,
        ...     web_results=web_matches,
        ...     intent=user_intent,
        ...     top_k=5,
        ...     use_rrf=True  # Enable RRF reranking
        ... )
        >>> print(len(docs))  # Top 5 results
        5
    """

    def __init__(self):
        """Initialize the hybrid scorer with fast reranking methods."""
        pass

    def score_and_rank(
        self,
        vector_results: List[Any],
        bm25_results: List[Any],
        web_results: List[Any],
        intent: IntentClassification,
        top_k: int = 5
    ) -> List[RetrievedDocument]:
        """
        Score all results using weighted combination and return top-k.

        Scoring formula:
            final_score = (vector_score * VECTOR_WEIGHT) + (bm25_score * BM25_WEIGHT) + WEB_BONUS

        Process:
        1. Convert vector results to RetrievedDocument (apply vector weight)
        2. Merge BM25 results (boost existing or add new with BM25 weight)
        3. Add web results (apply web bonus)
        4. Filter by framework/language metadata
        5. Sort by score and return top-k

        Args:
            vector_results: EmbeddingMatch objects from vector search
            bm25_results: (Document, score) tuples from BM25 search
            web_results: WebSearchResult objects from web API
            intent: User intent for metadata filtering
            top_k: Number of results to return

        Returns:
            List of RetrievedDocument objects, sorted by score descending

        Example:
            >>> vector_res = [EmbeddingMatch(...), ...]
            >>> bm25_res = [(Document(...), 0.8), ...]
            >>> web_res = [WebSearchResult(...), ...]
            >>> docs = scorer.score_and_rank(vector_res, bm25_res, web_res, intent, top_k=5)
            >>> docs[0].relevance_score > docs[1].relevance_score
            True
        """
        logger.debug(f"Scoring {len(vector_results)} vector + {len(bm25_results)} BM25 + {len(web_results)} web results")

        # Build unified document list
        all_docs = []

        # Add vector results (apply vector weight)
        for match in vector_results:
            doc = RetrievedDocument(
                file_path=match.document_path,
                title=match.document_title,
                framework=match.framework,
                language=match.language,
                topic=match.topic,
                granularity=match.granularity,
                matched_text=match.matched_text,
                section_title=match.section_title,
                section_level=match.section_level,
                content_block=match.content_block,
                sections=[],
                relevance_score=min(match.score * VECTOR_WEIGHT * 100, 100),  # Scale to 0-100
                retrieval_metadata={
                    "source": "vector",
                    "vector_score": match.score,
                }
            )
            all_docs.append(doc)

        # Add BM25 results (merge if document already exists from vector search)
        for bm25_doc, bm25_score in bm25_results:
            # Normalize BM25 score to 0-1 range (BM25 scores can be large)
            normalized_bm25 = min(bm25_score / 50.0, 1.0)  # Assume max BM25 score ~50

            # Check if document already in results from vector search
            existing = next(
                (d for d in all_docs if d.file_path == bm25_doc.file_path),
                None
            )

            if existing:
                # Boost existing document (combine scores)
                existing.relevance_score += min(normalized_bm25 * BM25_WEIGHT * 100, 100)
                existing.retrieval_metadata["bm25_score"] = normalized_bm25
                existing.retrieval_metadata["source"] = "vector+bm25"
                logger.debug(f"Boosted {bm25_doc.file_path} with BM25 score")
            else:
                # Add new document (BM25 only)
                doc = RetrievedDocument(
                    file_path=bm25_doc.file_path,
                    title=bm25_doc.title,
                    framework=bm25_doc.framework,
                    language=bm25_doc.language,
                    topic=bm25_doc.topic,
                    granularity="document",
                    matched_text=bm25_doc.sections[0].content if bm25_doc.sections else "",  # No truncation
                    section_title=None,
                    section_level=None,
                    content_block=None,
                    sections=[],
                    relevance_score=min(normalized_bm25 * BM25_WEIGHT * 100, 100),
                    retrieval_metadata={
                        "source": "bm25",
                        "bm25_score": normalized_bm25,
                    }
                )
                all_docs.append(doc)

        # Add web results (already deduplicated)
        for web_result in web_results:
            doc = RetrievedDocument(
                file_path=web_result.url,
                title=web_result.title,
                framework="web",
                language="unknown",
                topic="web_search",
                granularity="document",
                matched_text=web_result.content,  # No truncation
                section_title=None,
                section_level=None,
                content_block=None,
                sections=[],
                relevance_score=min(web_result.score * WEB_BONUS * 100, 100),  # Use as weight, not additive bonus
                retrieval_metadata={
                    "source": "web",
                    "url": web_result.url,
                    "web_score": web_result.score,
                }
            )
            all_docs.append(doc)

        logger.debug(f"Total documents before filtering: {len(all_docs)}")

        # Filter by framework/language metadata
        filtered = self._filter_by_metadata(all_docs, intent)

        # Sort by hybrid score
        ranked = sorted(filtered, key=lambda d: d.relevance_score, reverse=True)

        logger.info(f"Hybrid scoring: {len(all_docs)} docs → {len(filtered)} after filtering → returning top {top_k}")

        return ranked[:top_k]

    def _filter_by_metadata(
        self,
        docs: List[RetrievedDocument],
        intent: IntentClassification
    ) -> List[RetrievedDocument]:
        """
        Filter documents by framework and language metadata.

        This removes documents that don't match the user's intent,
        except for web results which are always kept (assumed relevant).

        Supports multiple frameworks/languages (e.g., ['langchain', 'langgraph']).

        Args:
            docs: List of documents to filter
            intent: User intent with framework/language preferences (can be lists)

        Returns:
            Filtered list of documents

        Example:
            >>> docs = [doc1, doc2, doc3]  # doc1: langchain, doc2: langgraph, doc3: web
            >>> intent.framework = ["langgraph"]
            >>> filtered = scorer._filter_by_metadata(docs, intent)
            >>> len(filtered)
            2  # doc2 (langgraph) + doc3 (web)
        """
        filtered = docs

        # Filter by framework (keep web results)
        if intent.framework:
            # Handle both single string and list
            frameworks = intent.framework if isinstance(intent.framework, list) else [intent.framework]
            # Remove "general" from filter (means no filtering)
            frameworks = [f for f in frameworks if f != "general"]

            if frameworks:
                filtered = [
                    d for d in filtered
                    if d.framework in frameworks or d.framework == "web"
                ]
                if len(filtered) < len(docs):
                    logger.debug(f"Filtered by frameworks {frameworks}: {len(docs)} → {len(filtered)}")

        # Filter by language (keep web results and docs with unknown language)
        if intent.language:
            # Handle both single string and list
            languages = intent.language if isinstance(intent.language, list) else [intent.language]

            filtered = [
                d for d in filtered
                if d.language in languages
                or d.language in [None, "unknown", ""]
                or d.framework == "web"
            ]
            if len(filtered) < len(docs):
                logger.debug(f"Filtered by languages {languages}: {len(docs)} → {len(filtered)}")

        return filtered

    def _add_results_to_registry(
        self,
        results: List[Any],
        doc_key_func: callable,
        doc_func: callable,
        source: str,
        k_value: int,
        doc_registry: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Helper method to add results to the document registry with RRF scoring.

        Args:
            results: List of result items to process
            doc_key_func: Function to extract document key from each result
            doc_func: Function to convert result to RetrievedDocument
            source: Source name (e.g., "vector", "bm25", "web")
            k_value: RRF k parameter for score calculation
            doc_registry: Document registry to update
        """
        for rank, result in enumerate(results, start=1):
            doc_key = doc_key_func(result)
            rrf_score = 1.0 / (k_value + rank)

            if doc_key not in doc_registry:
                doc_registry[doc_key] = {
                    "doc": doc_func(result),
                    "rrf_score": 0.0,
                    "sources": []
                }

            doc_registry[doc_key]["rrf_score"] += rrf_score
            doc_registry[doc_key]["sources"].append(source)

    def score_and_rank_rrf(
        self,
        vector_results: List[Any],
        bm25_results: List[Any],
        web_results: List[Any],
        intent: IntentClassification,
        top_k: int = 5,
        k: int = RRF_K_PARAMETER
    ) -> List[RetrievedDocument]:
        """
        Score and rank using Reciprocal Rank Fusion (RRF) with web result boosting.

        RRF is a fast, parameter-free ranking algorithm that combines rankings
        from multiple retrievers without needing score normalization.

        Formula: RRF_score(d) = Σ(1 / (k + rank_i(d)))
        where rank_i(d) is the rank of document d in retriever i.

        **Web Result Boosting:**
        Web results use k/3 instead of k to compensate for appearing in only one
        ranking list, while offline docs appear in both vector AND BM25. This ensures
        relevant web results aren't outranked by irrelevant offline docs that match
        only on keywords.

        Benefits over weighted combination:
        - No score normalization needed
        - Robust to score scale differences
        - Often outperforms weighted methods
        - Very fast (O(n) complexity)

        Args:
            vector_results: EmbeddingMatch objects from vector search
            bm25_results: (Document, score) tuples from BM25 search
            web_results: WebSearchResult objects from web API
            intent: User intent for metadata filtering
            top_k: Number of results to return
            k: RRF constant (default 60, standard value from literature)

        Returns:
            List of RetrievedDocument objects, ranked by RRF score

        Example:
            >>> scorer = HybridScorer()
            >>> docs = scorer.score_and_rank_rrf(vector_res, bm25_res, web_res, intent)
            >>> docs[0].relevance_score  # RRF score (scaled to 0-100)
            75.3
        """
        logger.debug(f"RRF scoring {len(vector_results)} vector + {len(bm25_results)} BM25 + {len(web_results)} web")

        # Build document registry with RRF scores
        doc_registry: Dict[str, Dict[str, Any]] = {}

        # Add vector results (rank-based scoring)
        self._add_results_to_registry(
            vector_results,
            lambda match: match.document_path,
            self._match_to_document,
            "vector",
            k,
            doc_registry
        )

        # Add BM25 results (rank-based scoring)
        self._add_results_to_registry(
            bm25_results,
            lambda item: item[0].file_path,  # item is (bm25_doc, score) tuple
            lambda item: self._bm25_to_document(item[0]),  # extract doc from tuple
            "bm25",
            k,
            doc_registry
        )

        # Add web results (rank-based scoring with boost)
        # Use lower k value for web results to boost their scores (they only appear in one list)
        # This compensates for offline docs appearing in both vector AND BM25
        web_k = k // 3  # Significantly lower k = higher scores for web results
        self._add_results_to_registry(
            web_results,
            lambda web_result: web_result.url,
            self._web_to_document,
            "web",
            web_k,  # Using web_k instead of k for boosting
            doc_registry
        )

        # Convert to list and update relevance scores
        all_docs = []
        for doc_data in doc_registry.values():
            doc = doc_data["doc"]
            # Scale RRF score to 0-100 for consistency
            doc.relevance_score = min(doc_data["rrf_score"] * 100 * k, 100.0)
            doc.retrieval_metadata["rrf_score"] = doc_data["rrf_score"]
            doc.retrieval_metadata["source"] = "+".join(doc_data["sources"])
            doc.retrieval_metadata["reranking_method"] = "rrf"
            all_docs.append(doc)

        # Filter and sort
        filtered = self._filter_by_metadata(all_docs, intent)
        ranked = sorted(filtered, key=lambda d: d.relevance_score, reverse=True)

        logger.debug(f"RRF scoring: {len(all_docs)} docs → {len(filtered)} after filtering → returning top {top_k}")

        return ranked[:top_k]

    def _match_to_document(self, match: Any) -> RetrievedDocument:
        """Convert EmbeddingMatch to RetrievedDocument."""
        return RetrievedDocument(
            file_path=match.document_path,
            title=match.document_title,
            framework=match.framework,
            language=match.language,
            topic=match.topic,
            granularity=match.granularity,
            matched_text=match.matched_text,
            section_title=match.section_title,
            section_level=match.section_level,
            content_block=match.content_block,
            sections=[],
            relevance_score=0.0,  # Will be set by RRF
            retrieval_metadata={"source": "vector"}
        )

    def _bm25_to_document(self, bm25_doc: Any) -> RetrievedDocument:
        """Convert BM25 Document to RetrievedDocument."""
        return RetrievedDocument(
            file_path=bm25_doc.file_path,
            title=bm25_doc.title,
            framework=bm25_doc.framework,
            language=bm25_doc.language,
            topic=bm25_doc.topic,
            granularity="document",
            matched_text=bm25_doc.sections[0].content if bm25_doc.sections else "",
            section_title=None,
            section_level=None,
            content_block=None,
            sections=[],
            relevance_score=0.0,  # Will be set by RRF
            retrieval_metadata={"source": "bm25"}
        )

    def _web_to_document(self, web_result: Any) -> RetrievedDocument:
        """Convert WebSearchResult to RetrievedDocument."""
        return RetrievedDocument(
            file_path=web_result.url,
            title=web_result.title,
            framework="web",
            language="unknown",
            topic="web_search",
            granularity="document",
            matched_text=web_result.content,
            section_title=None,
            section_level=None,
            content_block=None,
            sections=[],
            relevance_score=0.0,  # Will be set by RRF
            retrieval_metadata={"source": "web", "url": web_result.url}
        )
