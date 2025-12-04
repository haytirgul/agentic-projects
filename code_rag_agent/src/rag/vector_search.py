"""Vector database search using FAISS for efficient similarity search.

This module provides content-block-level granular retrieval using FAISS (Facebook AI Similarity Search)
for fast approximate nearest neighbor search.
"""

from typing import List, Dict, Optional, Literal, Any
from pathlib import Path
import logging
import json
from dataclasses import dataclass

# FAISS imports
try:
    import faiss
except ImportError:
    raise ImportError("FAISS is required. Install with: pip install faiss-cpu")

# Additional imports for FAISS
import pickle
import numpy as np

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
    Vector database search using FAISS for efficient similarity search.

    Key features:
    1. **3-level granularity**: Document, section, and content-block level embeddings
    2. **FAISS indexing**: Fast approximate nearest neighbor search with IVF or HNSW
    3. **Persistent storage**: FAISS index and metadata stored on disk
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
        Initialize vector search engine with FAISS.

        Args:
            persist_directory: Directory where FAISS index and metadata will be stored
            collection_name: Name of the collection (used for file naming)
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

        # FAISS file paths
        self.index_path = self.persist_directory / f"{collection_name}.faiss"
        self.metadata_path = self.persist_directory / f"{collection_name}_metadata.pkl"
        self.config_path = self.persist_directory / f"{collection_name}_config.pkl"

        # Initialize FAISS components
        logger.info(f"Initializing FAISS at {self.persist_directory}")

        # Setup embedding function
        self.embedding_function = self._create_embedding_function(
            embedding_model, google_api_key
        )

        # Load or create FAISS index
        self.index = None
        self.metadata = []
        self.id_to_idx = {}  # Maps document IDs to FAISS indices

        if self.index_path.exists() and self.metadata_path.exists():
            self._load_index()
        else:
            logger.info(f"Creating new FAISS index '{collection_name}'")
            self._create_empty_index()

    def _create_embedding_function(self, model_name: str, google_api_key: Optional[str]):
        """
        Create FAISS-compatible embedding function.

        Args:
            model_name: Model identifier
            google_api_key: API key for Google models

        Returns:
            Embedding function that returns numpy arrays
        """
        if model_name.startswith("models/"):
            # Google Gemini embeddings
            if google_api_key is None:
                google_api_key = GOOGLE_API_KEY

            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                logger.info(f"Using Google Gemini embeddings: {model_name}")
                return GoogleGenerativeAIEmbeddings(
                    model=model_name,
                    google_api_key=google_api_key
                )
            except ImportError:
                raise ImportError("langchain-google-genai is required for Google embeddings")
        else:
            # Hugging Face embeddings (sentence-transformers)
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                logger.info(f"Using Hugging Face embeddings: {model_name} (local)")
                return HuggingFaceEmbeddings(model_name=model_name)
            except ImportError:
                raise ImportError("langchain-huggingface is required for Hugging Face embeddings")

    def _create_empty_index(self):
        """Create an empty FAISS index."""
        # We'll create the index when we know the embedding dimension
        # For now, just initialize empty structures
        self.index = None
        self.metadata = []
        self.id_to_idx = {}

    def _load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_idx = data['id_to_idx']

            logger.info(f"Loaded FAISS index with {len(self.metadata)} embeddings")

        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}")
            self._create_empty_index()

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata
            data = {
                'metadata': self.metadata,
                'id_to_idx': self.id_to_idx
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(data, f)

            logger.debug(f"Saved FAISS index with {len(self.metadata)} embeddings")

    def build_index(self, documents: List[Any], force_rebuild: bool = False) -> int:
        """
        Build FAISS vector index from documents with 3-level granularity.

        Args:
            documents: List of Document objects
            force_rebuild: If True, clear existing index and rebuild from scratch

        Returns:
            Total number of embeddings created
        """

        if force_rebuild:
            logger.info("Force rebuild: Deleting existing FAISS index")
            self._create_empty_index()

        # Check if already indexed
        existing_count = len(self.metadata) if self.metadata else 0
        if existing_count > 0 and not force_rebuild:
            logger.info(f"Index already has {existing_count} embeddings. Use force_rebuild=True to rebuild.")
            return existing_count

        logger.info("Building FAISS vector index with 3-level granularity...")

        # Prepare data structures
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

        if not texts:
            logger.warning("No texts to embed")
            return 0

        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = []
        batch_size = 100  # Smaller batch size for embeddings

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                batch_embeddings = self.embedding_function.embed_documents(batch_texts)
                embeddings.extend(batch_embeddings)
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"  Generated embeddings for {i + batch_size}/{len(texts)} texts...")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch {i//batch_size}: {e}")
                # Add zero vectors as fallback
                embeddings.extend([[0.0] * 768] * len(batch_texts))  # Assuming 768-dim embeddings

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Create FAISS index
        dimension = embeddings_array.shape[1]
        logger.info(f"Creating FAISS index with dimension {dimension}")

        # Use IndexIVFFlat for larger datasets, IndexFlatIP for cosine similarity
        if len(embeddings) > 1000:
            # Use IVF with PQ for larger datasets
            nlist = min(100, max(4, len(embeddings) // 39))  # Rule of thumb: sqrt(n)/4
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)  # PQ with 8 bytes per vector
        else:
            # Use simple flat index with inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)

        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings_array)

        # Add vectors to index
        logger.info(f"Adding {len(embeddings_array)} vectors to FAISS index...")
        self.index.add(embeddings_array)

        # Store metadata and ID mapping
        self.metadata = metadatas
        self.id_to_idx = {ids[i]: i for i in range(len(ids))}

        # Save index and metadata
        self._save_index()

        # Log granularity breakdown
        granularity_counts = {}
        for meta in metadatas:
            level = meta['granularity']
            granularity_counts[level] = granularity_counts.get(level, 0) + 1

        logger.info(f"FAISS vector index built: {len(texts)} total embeddings")
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
        Search using FAISS vector similarity.

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
        if self.index is None or not self.metadata:
            logger.warning("FAISS index not loaded or empty")
            return []

        # Generate embedding for query
        try:
            query_embedding = self.embedding_function.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Search FAISS index
        # Request more results since we'll filter afterwards
        search_top_k = min(len(self.metadata), top_k * 5)  # Get more results for filtering

        try:
            # FAISS search returns distances and indices
            distances, indices = self.index.search(query_vector, search_top_k)
            similarity_scores = distances[0]  # FAISS returns inner product (cosine similarity)
            result_indices = indices[0]
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        # Parse results and apply metadata filtering
        matches = []
        for idx, similarity in zip(result_indices, similarity_scores):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            metadata = self.metadata[idx]

            # Apply metadata filters
            if framework:
                if isinstance(framework, list):
                    if metadata.get('framework') not in framework:
                        continue
                else:
                    if metadata.get('framework') != framework:
                        continue

            if language:
                if isinstance(language, list):
                    if metadata.get('language') not in language:
                        continue
                else:
                    if metadata.get('language') != language:
                        continue

            if topic and metadata.get('topic') != topic:
                continue

            # Parse content block if present
            content_block = None
            if metadata.get('content_block_content'):
                try:
                    content_block = json.loads(metadata['content_block_content'])
                except (json.JSONDecodeError, KeyError):
                    pass

            # For content_block matches, expand to full section content
            # This prevents incomplete/cut-off text in responses
            # Since FAISS doesn't store the text, we reconstruct it from metadata
            expanded_text = self._reconstruct_text_from_metadata(metadata)
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
                score=float(similarity),
                matched_text=expanded_text,
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

    def _reconstruct_text_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Reconstruct text content from metadata since FAISS doesn't store text.

        Args:
            metadata: Metadata dictionary from stored index

        Returns:
            Reconstructed text content
        """
        granularity = metadata.get('granularity', 'document')

        if granularity == 'content_block' and metadata.get('content_block_content'):
            # For content blocks, we stored the full block content as JSON
            try:
                block = json.loads(metadata['content_block_content'])
                return block.get('content', '')
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: reconstruct basic text from title and metadata
        title = metadata.get('document_title', '')
        section_title = metadata.get('section_title', '')

        if section_title and section_title != title:
            return f"{section_title}\n\nFrom document: {title}"
        else:
            return title

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
        return len(self.metadata) if self.metadata else 0

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index."""
        total_count = len(self.metadata) if self.metadata else 0

        granularity_counts = {}
        framework_counts = {}

        if self.metadata:
            for meta in self.metadata:
                # Count by granularity
                gran = meta.get('granularity', 'unknown')
                granularity_counts[gran] = granularity_counts.get(gran, 0) + 1

                # Count by framework
                fw = meta.get('framework', 'unknown')
                framework_counts[fw] = framework_counts.get(fw, 0) + 1

        index_info = {}
        if self.index is not None:
            index_info = {
                "index_type": type(self.index).__name__,
                "is_trained": getattr(self.index, 'is_trained', True),
                "dimension": getattr(self.index, 'd', 'unknown'),
            }

        return {
            "total_embeddings": total_count,
            "granularity_breakdown": granularity_counts,
            "framework_breakdown": framework_counts,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
            "index_info": index_info,
        }

    def reset(self) -> None:
        """Delete the FAISS index and reset."""
        logger.warning(f"Resetting FAISS index '{self.collection_name}'")

        # Remove index and metadata files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        if self.config_path.exists():
            self.config_path.unlink()

        # Reset in-memory structures
        self._create_empty_index()
