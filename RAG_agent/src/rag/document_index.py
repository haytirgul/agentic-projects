"""In-memory index of all knowledge base documents."""

from pathlib import Path
import json
from typing import Dict, List, Optional
from models.rag_models import Document, DocumentSection
import logging

logger = logging.getLogger(__name__)


class DocumentIndex:
    """In-memory index of all knowledge base documents."""

    def __init__(self, knowledge_base_path: Path):
        """
        Load all JSON files into memory.

        Args:
            knowledge_base_path: Path to json_files/ directory
        """
        self.knowledge_base_path = knowledge_base_path
        self.documents: Dict[str, Document] = {}
        self.by_framework: Dict[str, List[str]] = {}
        self.by_language: Dict[str, List[str]] = {}
        self.by_topic: Dict[str, List[str]] = {}

        self._load_all_documents()
        self._build_indexes()

        logger.info(f"Loaded {len(self.documents)} documents into memory")

    def _load_all_documents(self):
        """Scan directory and load all JSON files."""
        json_files = list(self.knowledge_base_path.rglob("*.json"))

        for file_path in json_files:
            try:
                relative_path = str(file_path.relative_to(self.knowledge_base_path))

                with file_path.open(encoding="utf-8") as f:
                    data = json.load(f)

                # Parse framework/language from path
                parts = Path(relative_path).parts
                framework = parts[0] if len(parts) > 0 else "unknown"

                # Determine language and topic based on framework
                if framework == "langchain":
                    # langchain/javascript|python/topic/filename
                    language = parts[1] if len(parts) > 1 else "unknown"
                    topic = parts[2] if len(parts) > 2 else "general"
                else:
                    # langgraph and langsmith: framework/topic/filename
                    language = "unknown"
                    topic = parts[1] if len(parts) > 1 else "general"

                # Extract sections recursively
                sections = self._extract_sections(data)

                # Create Document object
                doc = Document(
                    file_path=relative_path,
                    title=data.get("title", file_path.stem),
                    framework=framework,
                    language=language,
                    topic=topic,
                    sections=sections,
                    raw_data=data
                )

                self.documents[relative_path] = doc

            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

    def _extract_sections(self, node: dict, level: int = 1) -> List[DocumentSection]:
        """Recursively extract sections from JSON tree."""
        sections = []

        # Current node
        title = node.get("title", "")
        content_blocks = node.get("content_blocks", [])

        # Concatenate content blocks
        content_parts = []
        for block in content_blocks:
            block_type = block.get("type", "text")
            block_content = block.get("content", "")

            if block_type == "code":
                content_parts.append(f"```\n{block_content}\n```")
            else:
                content_parts.append(block_content)

        content = "\n\n".join(content_parts)

        if title and content:
            sections.append(DocumentSection(
                title=title,
                level=level,
                content=content,
                content_blocks=content_blocks
            ))

        # Recurse into children
        for child in node.get("children", []):
            sections.extend(self._extract_sections(child, level + 1))

        return sections

    def _build_indexes(self):
        """Build lookup indexes."""
        for path, doc in self.documents.items():
            # Framework index
            if doc.framework not in self.by_framework:
                self.by_framework[doc.framework] = []
            self.by_framework[doc.framework].append(path)

            # Language index
            if doc.language not in self.by_language:
                self.by_language[doc.language] = []
            self.by_language[doc.language].append(path)

            # Topic index
            if doc.topic not in self.by_topic:
                self.by_topic[doc.topic] = []
            self.by_topic[doc.topic].append(path)

    def filter(
        self,
        framework: Optional[str] = None,
        language: Optional[str] = None,
        topic: Optional[str] = None
    ) -> List[Document]:
        """
        Filter documents by criteria.

        Args:
            framework: Filter by framework (langchain, langgraph, langsmith)
            language: Filter by language (python, javascript)
            topic: Filter by topic directory (concepts, how-tos, etc.)

        Returns:
            List of matching documents
        """
        candidates = set(self.documents.keys())

        if framework:
            candidates &= set(self.by_framework.get(framework, []))

        if language:
            candidates &= set(self.by_language.get(language, []))

        if topic:
            candidates &= set(self.by_topic.get(topic, []))

        return [self.documents[path] for path in candidates]

    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        return list(self.documents.values())

    def get_document(self, file_path: str) -> Optional[Document]:
        """Get document by path."""
        return self.documents.get(file_path)
