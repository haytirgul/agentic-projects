"""BM25 keyword search over document corpus."""

from rank_bm25 import BM25Okapi
from typing import List, Tuple
import logging
from models.rag_models import Document

logger = logging.getLogger(__name__)


class BM25SearchEngine:
    """BM25 keyword search over document corpus."""

    def __init__(self, documents: List[Document]):
        """
        Build BM25 index over documents.

        Args:
            documents: List of all documents
        """
        self.documents = documents

        # Tokenize documents (simple whitespace + lowercase)
        self.corpus = []
        self.doc_map = []  # Map index → document

        for doc in documents:
            # Combine title + all section content
            text_parts = [doc.title]
            for section in doc.sections:
                text_parts.append(section.title)
                text_parts.append(section.content)

            full_text = " ".join(text_parts)
            tokens = full_text.lower().split()

            self.corpus.append(tokens)
            self.doc_map.append(doc)

        # Build BM25 index
        self.bm25 = BM25Okapi(self.corpus)

        logger.info(f"Built BM25 index over {len(self.corpus)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search documents using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Sort by score
        doc_scores = [(self.doc_map[i], scores[i]) for i in range(len(scores))]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return doc_scores[:top_k]
