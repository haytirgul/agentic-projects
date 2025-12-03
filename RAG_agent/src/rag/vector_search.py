"""Vector database search using ChromaDB with HNSW for efficient similarity search.

This module provides content-block-level granular retrieval using ChromaDB's HNSW
(Hierarchical Navigable Small World) algorithm for fast approximate nearest neighbor search.
"""

from typing import List, Dict, Optional, Literal, Any
from pathlib import Path
import logging
import json
import asyncio
from dataclasses import dataclass

# ChromaDB imports
from chromadb.config import Settings
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction, SentenceTransformerEmbeddingFunction

# Local imports
from settings import GOOGLE_API_KEY, DATA_DIR
from src.rag.document_index import DocumentIndex

logger = logging.getLogger(__name__)

# Type alias for granularity levels
GranularityLevel = Literal["document", "section", "content_block"]


@dataclass
class EmbeddingMatch:
    """Represents a match from vector search with granularity info."""

    document_path: str
    document_title: str
    framework: str
    language: str
    topic: str
    granularity: GranularityLevel
    score: float
    matched_text: str
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    content_block: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class VectorSearchEngine:
    """
    Vector database search using ChromaDB with HNSW indexing.

    Key features:
    1. **3-level granularity**: Document, section, and content-block level embeddings
    2. **HNSW indexing**: Fast approximate nearest neighbor search (better than cosine similarity)
    3. **Persistent storage**: ChromaDB persists to disk automatically
    4. **Metadata filtering**: Filter by framework, language, topic before semantic search
    5. **Granularity-based reranking**: Prefers finer-grained matches
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "documentation_embeddings",
        embedding_model: str = "models/embedding-001",
        google_api_key: Optional[str] = None,
        content_preview_chars: int = 300,
        enable_block_level: bool = True,
        document_index: Optional['DocumentIndex'] = None,
    ):
        """
        Initialize vector search engine with ChromaDB.

        Args:
            persist_directory: Directory where ChromaDB will persist data
            collection_name: Name of the ChromaDB collection
            embedding_model: Embedding model name (Google Gemini or Hugging Face)
            google_api_key: Google API key (only needed for Google models)
            content_preview_chars: Max characters per content block (default: 300)
            enable_block_level: Whether to enable content-block-level embeddings
            document_index: Optional pre-loaded DocumentIndex (avoids reloading during search)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.content_preview_chars = content_preview_chars
        self.enable_block_level = enable_block_level
        self.document_index = document_index

        # Import ChromaDB
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required. Install with: pip install chromadb"
            )

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        # Create directory if it doesn't exist (but don't delete if it does!)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

        # Setup embedding function
        self.embedding_function = self._create_embedding_function(
            embedding_model, google_api_key
        )

        # Get or create collection with HNSW indexing
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
            )
            logger.info(f"Loaded existing collection '{collection_name}' with {self.collection.count()} embeddings")
        except Exception:
            logger.info(f"Creating new collection '{collection_name}' with HNSW indexing")
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},  # Use cosine distance for HNSW
            )

    def _create_embedding_function(self, model_name: str, google_api_key: Optional[str]):
        """
        Create ChromaDB-compatible embedding function.

        Args:
            model_name: Model identifier
            google_api_key: API key for Google models

        Returns:
            ChromaDB embedding function
        """
        if model_name.startswith("models/"):
            # Google Gemini embeddings
            if google_api_key is None:
                google_api_key = GOOGLE_API_KEY

            logger.info(f"Using Google Gemini embeddings: {model_name}")
            return GoogleGenerativeAiEmbeddingFunction(
                api_key=google_api_key,
                model_name=model_name,
            )
        else:
            # Hugging Face embeddings (sentence-transformers)
            logger.info(f"Using Hugging Face embeddings: {model_name} (local)")
            return SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )

    def build_index(self, documents: List[Any], force_rebuild: bool = False) -> int:
        """
        Build vector index from documents with 3-level granularity.

        Args:
            documents: List of Document objects
            force_rebuild: If True, clear existing collection and rebuild from scratch

        Returns:
            Total number of embeddings created
        """

        if force_rebuild:
            logger.info("Force rebuild: Deleting existing collection")
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        # Check if already indexed
        existing_count = self.collection.count()
        if existing_count > 0 and not force_rebuild:
            logger.info(f"Collection already has {existing_count} embeddings. Use force_rebuild=True to rebuild.")
            return existing_count

        logger.info("Building vector index with 3-level granularity...")

        # Prepare batches for ChromaDB
        texts = []
        metadatas = []
        ids = []
        id_counter = 0

        for doc in documents:
            # Level 1: Document-level embedding
            if doc.sections:
                first_section_preview = doc.sections[0].content[:self.content_preview_chars]
                doc_text = f"{doc.title}\n\n{first_section_preview}"
            else:
                doc_text = doc.title

            texts.append(doc_text)
            metadatas.append({
                "granularity": "document",
                "document_path": doc.file_path,
                "document_title": doc.title,
                "framework": doc.framework,
                "language": doc.language,
                "topic": doc.topic,
                "section_title": "",
                "section_level": 0,
                "content_block_type": "",
            })
            ids.append(f"doc_{id_counter}")
            id_counter += 1

            # Level 2 & 3: Section and content-block embeddings
            for section in doc.sections:
                # Level 2: Section-level embedding
                section_preview = section.content[:self.content_preview_chars]
                section_text = f"{section.title}\n\n{section_preview}"

                texts.append(section_text)
                metadatas.append({
                    "granularity": "section",
                    "document_path": doc.file_path,
                    "document_title": doc.title,
                    "framework": doc.framework,
                    "language": doc.language,
                    "topic": doc.topic,
                    "section_title": section.title,
                    "section_level": section.level,
                    "content_block_type": "",
                })
                ids.append(f"sec_{id_counter}")
                id_counter += 1

                # Level 3: Content-block-level embeddings
                if self.enable_block_level and section.content_blocks:
                    for block_idx, block in enumerate(section.content_blocks):
                        # Only embed text blocks with sufficient content
                        if block.get('type') == 'text' and len(block.get('content', '')) > 50:
                            block_content = block['content'][:self.content_preview_chars]
                            block_text = f"{section.title}\n\n{block_content}"

                            texts.append(block_text)
                            metadatas.append({
                                "granularity": "content_block",
                                "document_path": doc.file_path,
                                "document_title": doc.title,
                                "framework": doc.framework,
                                "language": doc.language,
                                "topic": doc.topic,
                                "section_title": section.title,
                                "section_level": section.level,
                                "content_block_type": block.get('type', 'text'),
                                "content_block_index": block_idx,
                                # Store full block content for retrieval
                                "content_block_content": json.dumps(block),
                            })
                            ids.append(f"blk_{id_counter}")
                            id_counter += 1

        # Add to ChromaDB in batches (ChromaDB recommends batches of ~100-1000)
        batch_size = 500
        logger.info(f"Adding {len(texts)} embeddings to ChromaDB in batches of {batch_size}...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )

            if (i + batch_size) % 2000 == 0:
                logger.info(f"  Added {i + batch_size}/{len(texts)} embeddings...")

        # Log granularity breakdown
        granularity_counts = {}
        for meta in metadatas:
            level = meta['granularity']
            granularity_counts[level] = granularity_counts.get(level, 0) + 1

        logger.info(f"Vector index built: {len(texts)} total embeddings")
        logger.info(f"  Document-level: {granularity_counts.get('document', 0)}")
        logger.info(f"  Section-level: {granularity_counts.get('section', 0)}")
        logger.info(f"  Content-block-level: {granularity_counts.get('content_block', 0)}")

        return len(texts)

    def search(
        self,
        query: str,
        top_k: int = 5,
        framework: str | List[str] | None = None,
        language: str | List[str] | None = None,
        topic: Optional[str] = None,
        granularity_boost: bool = True,
    ) -> List[EmbeddingMatch]:
        """
        Search using HNSW-based vector similarity.

        Args:
            query: Search query
            top_k: Number of results to return (after reranking)
            framework: Optional framework filter (string or list: ['langchain', 'langgraph'])
            language: Optional language filter (string or list: ['python', 'javascript'])
            topic: Optional topic filter
            granularity_boost: If True, boost finer-grained matches

        Returns:
            List of EmbeddingMatch objects, sorted by boosted score descending
        """
        # Build metadata filter (ChromaDB requires explicit operators for multiple filters)
        where_filter = None
        filters = []

        # Handle framework filter (string or list)
        if framework:
            if isinstance(framework, list):
                if len(framework) == 1:
                    filters.append({"framework": framework[0]})
                else:
                    # Multiple frameworks: use $or operator
                    filters.append({"$or": [{"framework": f} for f in framework]})
            else:
                filters.append({"framework": framework})

        # Handle language filter (string or list)
        if language:
            if isinstance(language, list):
                if len(language) == 1:
                    filters.append({"language": language[0]})
                else:
                    # Multiple languages: use $or operator
                    filters.append({"$or": [{"language": lang} for lang in language]})
            else:
                filters.append({"language": language})

        if topic:
            filters.append({"topic": topic})

        # Build proper ChromaDB filter
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        # Query ChromaDB with HNSW
        # Request more results if we're doing granularity boosting
        query_top_k = top_k * 3 if granularity_boost else top_k

        results = self.collection.query(
            query_texts=[query],
            n_results=query_top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Parse results
        matches = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                document_text = results['documents'][0][i]
                distance = results['distances'][0][i]

                # Convert distance to similarity score (ChromaDB uses cosine distance)
                # Cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
                similarity = 1.0 - distance

                # Parse content block if present
                content_block = None
                if metadata.get('content_block_content'):
                    try:
                        content_block = json.loads(metadata['content_block_content'])
                    except:
                        pass

                # For content_block matches, expand to full section content
                # This prevents incomplete/cut-off text in responses
                expanded_text = document_text
                if metadata['granularity'] == 'content_block':
                    # Load the full section with all content blocks
                    full_section_content = self._load_full_section(
                        document_path=metadata['document_path'],
                        section_title=metadata.get('section_title'),
                    )
                    if full_section_content:
                        expanded_text = full_section_content
                        logger.debug(f"Expanded content_block to full section ({len(expanded_text)} chars)")

                match = EmbeddingMatch(
                    document_path=metadata['document_path'],
                    document_title=metadata['document_title'],
                    framework=metadata['framework'],
                    language=metadata['language'],
                    topic=metadata['topic'],
                    granularity=metadata['granularity'],
                    score=similarity,
                    matched_text=expanded_text,  # Use expanded text instead of truncated preview
                    section_title=metadata.get('section_title') or None,
                    section_level=metadata.get('section_level') or None,
                    content_block=content_block,
                    metadata=metadata,
                )
                matches.append(match)

        # Apply granularity-based reranking
        if granularity_boost:
            matches = self._rerank_by_granularity(matches)

        # Deduplicate by section
        # When multiple content_blocks from the same section match, keep only the best one
        matches = self._deduplicate_by_section(matches)

        return matches[:top_k]

    def _rerank_by_granularity(self, matches: List[EmbeddingMatch]) -> List[EmbeddingMatch]:
        """
        Rerank matches with granularity-based boosting.

        Content-block matches get highest boost, then sections, then documents.

        Args:
            matches: List of matches from vector search

        Returns:
            Reranked matches with boosted scores
        """
        # Granularity boost factors
        BOOST_FACTORS = {
            "content_block": 1.5,  # 50% boost for content blocks
            "section": 1.2,         # 20% boost for sections
            "document": 1.0,        # No boost for documents
        }

        for match in matches:
            boost = BOOST_FACTORS.get(match.granularity, 1.0)
            match.score *= boost

        # Re-sort by boosted scores
        matches.sort(key=lambda m: m.score, reverse=True)

        logger.debug(f"Reranked {len(matches)} matches with granularity boosting")
        return matches

    def _deduplicate_by_section(self, matches: List[EmbeddingMatch]) -> List[EmbeddingMatch]:
        """
        Deduplicate matches by (document_path, section_title).

        When multiple content_blocks from the same section match (because we expanded them
        to full sections), keep only the highest-scoring one. This prevents duplicate
        content in the final results.

        Args:
            matches: List of matches (already sorted by score descending)

        Returns:
            Deduplicated matches
        """
        seen_sections = set()
        deduplicated = []

        for match in matches:
            # Create unique key for (document, section)
            # For content_block and section granularity, we expanded to full section
            section_key = None
            if match.granularity in ["content_block", "section"] and match.section_title:
                section_key = (match.document_path, match.section_title)
            elif match.granularity == "document":
                # Document-level matches are unique by document
                section_key = (match.document_path, "__document__")

            if section_key:
                if section_key not in seen_sections:
                    seen_sections.add(section_key)
                    deduplicated.append(match)
                else:
                    logger.debug(f"Skipping duplicate section: {match.section_title} from {match.document_path}")
            else:
                # No section key, include it
                deduplicated.append(match)

        if len(deduplicated) < len(matches):
            logger.info(f"Deduplicated {len(matches)} matches to {len(deduplicated)} (removed {len(matches) - len(deduplicated)} duplicates)")

        return deduplicated

    def _load_full_section(self, document_path: str, section_title: Optional[str]) -> Optional[str]:
        """
        Load the full section content using the DocumentIndex.

        When a content_block matches, we return the ENTIRE section
        (all content blocks concatenated) instead of just the matched block.
        This prevents incomplete/cut-off text in responses.

        Args:
            document_path: Path to the JSON document (e.g., "langchain/python/concepts/memory.json")
            section_title: Title of the section to load

        Returns:
            Full section content with all blocks concatenated, or None if not found
        """
        if not section_title:
            return None

        try:
            # Get document from DocumentIndex if available
            if self.document_index is None:
                # Lazy load DocumentIndex (only if not provided during init)
                logger.warning("DocumentIndex not provided, lazy loading (this is slow!)")
                kb_path = DATA_DIR / "output" / "json_files"
                self.document_index = DocumentIndex(kb_path)
                logger.debug(f"Loaded DocumentIndex with {len(self.document_index.documents)} documents")

            # Get the document
            document = self.document_index.get_document(document_path)
            if not document:
                logger.warning(f"Document not found: {document_path}")
                return None

            # Find the matching section and return its full content
            for section in document.sections:
                if section.title == section_title:
                    return section.content

            logger.debug(f"Section '{section_title}' not found in {document_path}")
            return None

        except Exception as e:
            logger.error(f"Error loading full section from {document_path}: {e}")
            return None

    def get_collection_count(self) -> int:
        """Get the total number of embeddings in the collection."""
        return self.collection.count()

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        total_count = self.collection.count()

        # Query to count by granularity
        all_results = self.collection.get(
            include=["metadatas"],
        )

        granularity_counts = {}
        framework_counts = {}

        if all_results['metadatas']:
            for meta in all_results['metadatas']:
                # Count by granularity
                gran = meta.get('granularity', 'unknown')
                granularity_counts[gran] = granularity_counts.get(gran, 0) + 1

                # Count by framework
                fw = meta.get('framework', 'unknown')
                framework_counts[fw] = framework_counts.get(fw, 0) + 1

        return {
            "total_embeddings": total_count,
            "granularity_breakdown": granularity_counts,
            "framework_breakdown": framework_counts,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }

    def reset(self) -> None:
        """Delete the collection and reset."""
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
