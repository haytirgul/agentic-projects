"""Tests for retrieval components (no LLM in the loop).

This module tests the retrieval pipeline components:
- MetadataFilter: source_type/file_pattern filtering (soft folder filtering)
- HybridRetriever: folder boost scoring
- ContextExpander: parent/sibling/child expansion

Author: Hay Hoffman
Version: 1.0
"""

import pytest
from models.chunk import CodeChunk, MarkdownChunk, TextChunk
from models.retrieval import ExpandedChunk, RetrievalRequest, RetrievedChunk
from src.retrieval.context_expander import ContextExpander
from src.retrieval.metadata_filter import ChunkMetadata, MetadataFilter


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_code_metadata() -> list[ChunkMetadata]:
    """Sample code chunk metadata for tests."""
    return [
        ChunkMetadata(
            id="code_1",
            source_type="code",
            filename="client.py",
            file_path="httpx/_client.py",
            chunk_type="class",
            name="Client",
            parent_context="",
            start_line=10,
            end_line=100,
        ),
        ChunkMetadata(
            id="code_2",
            source_type="code",
            filename="client.py",
            file_path="httpx/_client.py",
            chunk_type="function",
            name="get",
            parent_context="Client",
            start_line=50,
            end_line=70,
        ),
        ChunkMetadata(
            id="code_3",
            source_type="code",
            filename="client.py",
            file_path="httpx/_client.py",
            chunk_type="function",
            name="post",
            parent_context="Client",
            start_line=75,
            end_line=95,
        ),
        ChunkMetadata(
            id="code_4",
            source_type="code",
            filename="base.py",
            file_path="httpx/_transports/base.py",
            chunk_type="class",
            name="BaseTransport",
            parent_context="",
            start_line=1,
            end_line=50,
        ),
        ChunkMetadata(
            id="code_5",
            source_type="code",
            filename="http11.py",
            file_path="httpx/_transports/http11.py",
            chunk_type="function",
            name="handle_request",
            parent_context="HTTP11Transport",
            start_line=20,
            end_line=80,
        ),
    ]


@pytest.fixture
def sample_markdown_metadata() -> list[ChunkMetadata]:
    """Sample markdown chunk metadata for tests."""
    return [
        ChunkMetadata(
            id="md_1",
            source_type="markdown",
            filename="README.md",
            file_path="docs/README.md",
            chunk_type="markdown_section",
            name="Installation",
            heading_level=1,
            start_line=1,
            end_line=20,
        ),
        ChunkMetadata(
            id="md_2",
            source_type="markdown",
            filename="README.md",
            file_path="docs/README.md",
            chunk_type="markdown_section",
            name="Quick Start",
            heading_level=2,
            start_line=5,
            end_line=15,
        ),
        ChunkMetadata(
            id="md_3",
            source_type="markdown",
            filename="api.md",
            file_path="docs/api.md",
            chunk_type="markdown_section",
            name="API Reference",
            heading_level=1,
            start_line=1,
            end_line=100,
        ),
    ]


@pytest.fixture
def sample_text_metadata() -> list[ChunkMetadata]:
    """Sample text chunk metadata for tests."""
    return [
        ChunkMetadata(
            id="text_1",
            source_type="text",
            filename="pyproject.toml",
            file_path="pyproject.toml",
            chunk_type="text_file",
            name="pyproject.toml",
            start_line=1,
            end_line=50,
        ),
        ChunkMetadata(
            id="text_2",
            source_type="text",
            filename="requirements.txt",
            file_path="requirements.txt",
            chunk_type="text_file",
            name="requirements.txt",
            start_line=1,
            end_line=20,
        ),
    ]


@pytest.fixture
def metadata_index(
    sample_code_metadata: list[ChunkMetadata],
    sample_markdown_metadata: list[ChunkMetadata],
    sample_text_metadata: list[ChunkMetadata],
) -> dict[str, list[ChunkMetadata]]:
    """Combined metadata index for all source types."""
    return {
        "code": sample_code_metadata,
        "markdown": sample_markdown_metadata,
        "text": sample_text_metadata,
    }


@pytest.fixture
def sample_code_chunks() -> dict[str, CodeChunk]:
    """Sample code chunks for context expansion tests."""
    return {
        "code_1": CodeChunk(
            id="code_1",
            source_type="code",
            chunk_type="class",
            content="class Client:\n    '''HTTP Client class'''\n    def __init__(self): pass",
            token_count=50,
            file_path="httpx/_client.py",
            filename="client.py",
            start_line=10,
            end_line=100,
            name="Client",
            full_name="Client",
            parent_context="",
            docstring="HTTP Client class",
        ),
        "code_2": CodeChunk(
            id="code_2",
            source_type="code",
            chunk_type="function",
            content="def get(self, url): return self.request('GET', url)",
            token_count=20,
            file_path="httpx/_client.py",
            filename="client.py",
            start_line=50,
            end_line=70,
            name="get",
            full_name="Client.get",
            parent_context="Client",
            docstring="Send GET request",
        ),
        "code_3": CodeChunk(
            id="code_3",
            source_type="code",
            chunk_type="function",
            content="def post(self, url, data): return self.request('POST', url, data=data)",
            token_count=25,
            file_path="httpx/_client.py",
            filename="client.py",
            start_line=75,
            end_line=95,
            name="post",
            full_name="Client.post",
            parent_context="Client",
            docstring="Send POST request",
        ),
    }


@pytest.fixture
def sample_markdown_chunks() -> dict[str, MarkdownChunk]:
    """Sample markdown chunks for context expansion tests."""
    return {
        "md_1": MarkdownChunk(
            id="md_1",
            source_type="markdown",
            chunk_type="markdown_section",
            content="# Installation\n\nInstall httpx using pip.\n\n## Quick Start\n\nBasic usage example.",
            token_count=30,
            file_path="docs/README.md",
            filename="README.md",
            start_line=1,
            end_line=20,
            name="Installation",
            full_name="Installation",
            parent_context="",
            heading_level=1,
        ),
        "md_2": MarkdownChunk(
            id="md_2",
            source_type="markdown",
            chunk_type="markdown_section",
            content="## Quick Start\n\nBasic usage example.",
            token_count=15,
            file_path="docs/README.md",
            filename="README.md",
            start_line=5,
            end_line=15,
            name="Quick Start",
            full_name="Installation > Quick Start",
            parent_context="Installation",
            heading_level=2,
        ),
    }


@pytest.fixture
def sample_text_chunks() -> dict[str, TextChunk]:
    """Sample text chunks for tests."""
    return {
        "text_1": TextChunk(
            id="text_1",
            source_type="text",
            chunk_type="text_file",
            content="[project]\nname = 'httpx'\nversion = '1.0.0'",
            token_count=20,
            file_path="pyproject.toml",
            filename="pyproject.toml",
            start_line=1,
            end_line=50,
            name="pyproject.toml",
            full_name="pyproject.toml",
            parent_context="",
            file_extension=".toml",
        ),
    }


class MockChunkLoader:
    """Mock chunk loader for testing (no disk I/O)."""

    def __init__(self, chunks: dict):
        """Initialize with pre-loaded chunks."""
        self.chunk_cache = chunks

    def load_chunk(self, chunk_id: str):
        """Load chunk by ID from cache."""
        if chunk_id not in self.chunk_cache:
            raise KeyError(f"Chunk {chunk_id} not found")
        return self.chunk_cache[chunk_id]


# =============================================================================
# MetadataFilter Tests
# =============================================================================


class TestMetadataFilter:
    """Tests for MetadataFilter (v1.2 - soft folder filtering)."""

    def test_filter_by_single_source_type(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test filtering by single source_type returns correct chunks."""
        request = RetrievalRequest(
            query="HTTP client implementation",
            source_types=["code"],
            reasoning="Need code implementation",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # Should return all code chunks
        assert len(result) == 5
        assert "code_1" in result
        assert "code_2" in result
        assert "md_1" not in result  # markdown excluded
        assert "text_1" not in result  # text excluded

    def test_filter_by_multiple_source_types(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test filtering by multiple source_types."""
        request = RetrievalRequest(
            query="Installation documentation and config",
            source_types=["markdown", "text"],
            reasoning="Need docs and config files",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # Should return markdown + text chunks
        assert len(result) == 5  # 3 markdown + 2 text
        assert "md_1" in result
        assert "text_1" in result
        assert "code_1" not in result  # code excluded

    def test_filter_by_file_patterns(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test filtering by file patterns (hard filter)."""
        request = RetrievalRequest(
            query="client code",
            source_types=["code"],
            file_patterns=["client.py"],
            reasoning="Need client code",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # Should return only chunks from client.py
        assert len(result) == 3
        assert "code_1" in result
        assert "code_2" in result
        assert "code_3" in result
        assert "code_4" not in result  # base.py excluded
        assert "code_5" not in result  # http11.py excluded

    def test_filter_by_file_patterns_wildcard(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test filtering by file patterns with wildcards."""
        request = RetrievalRequest(
            query="HTTP transport code",
            source_types=["code"],
            file_patterns=["*.py"],
            reasoning="Need Python files",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # Should return all Python files
        assert len(result) == 5

    def test_folder_filter_does_not_exclude(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test that folder filter does NOT exclude chunks (v1.2 soft filtering)."""
        request = RetrievalRequest(
            query="transport implementation",
            source_types=["code"],
            folders=["httpx/_transports/"],  # Only transport folder specified
            reasoning="Need transport code",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # v1.2: Folders do NOT exclude - all code chunks should be returned
        assert len(result) == 5
        assert "code_1" in result  # httpx/_client.py - NOT in folders, but still included
        assert "code_4" in result  # httpx/_transports/base.py - in folders
        assert "code_5" in result  # httpx/_transports/http11.py - in folders

    def test_empty_file_patterns_returns_all(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test that empty file_patterns returns all chunks for source_type."""
        request = RetrievalRequest(
            query="all markdown",
            source_types=["markdown"],
            file_patterns=[],  # Empty - no filtering
            reasoning="Need all markdown",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        assert len(result) == 3  # All markdown chunks

    def test_combined_filters(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test combining source_types and file_patterns filters."""
        request = RetrievalRequest(
            query="README documentation",
            source_types=["markdown"],
            file_patterns=["README.md"],
            reasoning="Need README",
        )

        filter_instance = MetadataFilter(metadata_index, request)
        result = filter_instance.apply()

        # Should return only README.md markdown chunks
        assert len(result) == 2
        assert "md_1" in result
        assert "md_2" in result
        assert "md_3" not in result  # api.md excluded

    def test_no_matching_source_type(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test filtering with non-existent source_type in index."""
        # Create index without 'text' type
        partial_index = {"code": metadata_index["code"]}

        request = RetrievalRequest(
            query="config files",
            source_types=["text"],
            reasoning="Need config",
        )

        filter_instance = MetadataFilter(partial_index, request)
        result = filter_instance.apply()

        # Should return empty list
        assert len(result) == 0


# =============================================================================
# Folder Boost Tests (HybridRetriever._apply_folder_boost)
# =============================================================================


class TestFolderBoost:
    """Tests for folder boost scoring in HybridRetriever."""

    def test_folder_boost_applied_to_matching_chunks(self):
        """Test that folder boost multiplies score for matching chunks."""
        # Simulate ranked results from RRF
        ranked_ids = [
            ("chunk1", 0.5),  # httpx/_client.py - NOT in transports
            ("chunk2", 0.4),  # httpx/_transports/base.py - IN transports
            ("chunk3", 0.3),  # httpx/_transports/http11.py - IN transports
        ]

        chunk_file_paths = {
            "chunk1": "httpx/_client.py",
            "chunk2": "httpx/_transports/base.py",
            "chunk3": "httpx/_transports/http11.py",
        }

        folders = ["httpx/_transports/"]
        boost_factor = 1.3

        # Apply folder boost logic (same as HybridRetriever._apply_folder_boost)
        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # chunk2 should now be ranked higher: 0.4 * 1.3 = 0.52 > 0.5
        assert boosted_results[0][0] == "chunk2"
        assert boosted_results[0][1] == pytest.approx(0.52, rel=1e-3)

        # chunk1 should be second (unchanged at 0.5)
        assert boosted_results[1][0] == "chunk1"
        assert boosted_results[1][1] == 0.5

        # chunk3 should be third: 0.3 * 1.3 = 0.39
        assert boosted_results[2][0] == "chunk3"
        assert boosted_results[2][1] == pytest.approx(0.39, rel=1e-3)

    def test_folder_boost_no_matching_folders(self):
        """Test that no boost is applied when no folders match."""
        ranked_ids = [
            ("chunk1", 0.5),
            ("chunk2", 0.4),
        ]

        chunk_file_paths = {
            "chunk1": "httpx/_client.py",
            "chunk2": "httpx/_models.py",
        }

        folders = ["httpx/_transports/"]  # Neither chunk matches
        boost_factor = 1.3

        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # Order should be unchanged, scores unchanged
        assert boosted_results[0] == ("chunk1", 0.5)
        assert boosted_results[1] == ("chunk2", 0.4)

    def test_folder_boost_multiple_folders(self):
        """Test boost with multiple target folders."""
        ranked_ids = [
            ("chunk1", 0.5),  # httpx/_client.py
            ("chunk2", 0.4),  # httpx/_transports/base.py
            ("chunk3", 0.3),  # docs/api.md
        ]

        chunk_file_paths = {
            "chunk1": "httpx/_client.py",
            "chunk2": "httpx/_transports/base.py",
            "chunk3": "docs/api.md",
        }

        folders = ["httpx/_transports/", "docs/"]
        boost_factor = 1.3

        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # chunk2: 0.4 * 1.3 = 0.52 (matches httpx/_transports/)
        # chunk1: 0.5 (no match)
        # chunk3: 0.3 * 1.3 = 0.39 (matches docs/)
        assert boosted_results[0][0] == "chunk2"
        assert boosted_results[0][1] == pytest.approx(0.52, rel=1e-3)
        assert boosted_results[1][0] == "chunk1"
        assert boosted_results[1][1] == 0.5
        assert boosted_results[2][0] == "chunk3"
        assert boosted_results[2][1] == pytest.approx(0.39, rel=1e-3)

    def test_folder_boost_empty_folders_list(self):
        """Test that empty folders list leaves scores unchanged."""
        ranked_ids = [
            ("chunk1", 0.5),
            ("chunk2", 0.4),
        ]

        chunk_file_paths = {
            "chunk1": "httpx/_client.py",
            "chunk2": "httpx/_transports/base.py",
        }

        folders = []  # Empty - no boost
        boost_factor = 1.3

        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # No change
        assert boosted_results[0] == ("chunk1", 0.5)
        assert boosted_results[1] == ("chunk2", 0.4)

    def test_folder_boost_preserves_non_matching_chunks(self):
        """Test that non-matching chunks are NOT excluded, just not boosted."""
        ranked_ids = [
            ("chunk1", 0.6),  # NOT in target folder
            ("chunk2", 0.3),  # IN target folder
        ]

        chunk_file_paths = {
            "chunk1": "httpx/_client.py",
            "chunk2": "httpx/_transports/base.py",
        }

        folders = ["httpx/_transports/"]
        boost_factor = 1.3

        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # chunk1 still ranked first (0.6 > 0.39)
        # chunk2 boosted but still lower: 0.3 * 1.3 = 0.39
        assert boosted_results[0][0] == "chunk1"
        assert boosted_results[0][1] == 0.6
        assert boosted_results[1][0] == "chunk2"
        assert boosted_results[1][1] == pytest.approx(0.39, rel=1e-3)


# =============================================================================
# Context Expansion Tests
# =============================================================================


class TestContextExpander:
    """Tests for ContextExpander (parent/sibling/child expansion)."""

    def test_expand_code_chunk_finds_parent_class(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test that code chunk expansion finds parent class."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create retrieved chunk for method
        method_chunk = sample_code_chunks["code_2"]  # Client.get
        retrieved = RetrievedChunk(
            chunk=method_chunk,
            rrf_score=0.5,
            request_reasoning="Need GET method",
            source_query="HTTP GET request",
        )

        expanded = expander.expand(retrieved)

        assert expanded is not None
        assert expanded.base_chunk.id == "code_2"
        assert expanded.rrf_score == 0.5

        # Should find parent class
        assert expanded.parent_chunk is not None
        assert expanded.parent_chunk.name == "Client"
        assert expanded.expansion_metadata["parent_added"] is True

    def test_expand_code_chunk_finds_sibling_methods(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test that code chunk expansion finds sibling methods."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create retrieved chunk for method
        method_chunk = sample_code_chunks["code_2"]  # Client.get
        retrieved = RetrievedChunk(
            chunk=method_chunk,
            rrf_score=0.5,
            request_reasoning="Need GET method",
            source_query="HTTP GET request",
        )

        expanded = expander.expand(retrieved)

        assert expanded is not None

        # Should find sibling method (post)
        sibling_names = [s.name for s in expanded.sibling_chunks]
        assert "post" in sibling_names
        assert expanded.expansion_metadata["siblings_added"] >= 1

    def test_expand_code_chunk_class_no_parent(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test that class chunks have no parent."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create retrieved chunk for class
        class_chunk = sample_code_chunks["code_1"]  # Client class
        retrieved = RetrievedChunk(
            chunk=class_chunk,
            rrf_score=0.6,
            request_reasoning="Need Client class",
            source_query="HTTP Client",
        )

        expanded = expander.expand(retrieved)

        assert expanded is not None
        assert expanded.parent_chunk is None  # Classes have no parent
        assert expanded.expansion_metadata["parent_added"] is False

    def test_expand_markdown_chunk_finds_children(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_markdown_chunks: dict[str, MarkdownChunk],
    ):
        """Test that markdown chunk expansion finds child sections."""
        chunk_loader = MockChunkLoader(sample_markdown_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create retrieved chunk for parent section
        parent_section = sample_markdown_chunks["md_1"]  # Installation (h1)
        retrieved = RetrievedChunk(
            chunk=parent_section,
            rrf_score=0.7,
            request_reasoning="Need installation docs",
            source_query="Install httpx",
        )

        expanded = expander.expand(retrieved)

        assert expanded is not None
        assert expanded.base_chunk.id == "md_1"

        # Should find child section (Quick Start)
        child_names = [c.name for c in expanded.child_sections]
        assert "Quick Start" in child_names
        assert expanded.expansion_metadata["children_added"] >= 1

    def test_expand_text_chunk_no_expansion(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_text_chunks: dict[str, TextChunk],
    ):
        """Test that text chunks have no expansion (self-contained)."""
        chunk_loader = MockChunkLoader(sample_text_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create retrieved chunk for text file
        text_chunk = sample_text_chunks["text_1"]  # pyproject.toml
        retrieved = RetrievedChunk(
            chunk=text_chunk,
            rrf_score=0.4,
            request_reasoning="Need project config",
            source_query="project configuration",
        )

        expanded = expander.expand(retrieved)

        assert expanded is not None
        assert expanded.parent_chunk is None
        assert len(expanded.sibling_chunks) == 0
        assert len(expanded.child_sections) == 0
        assert expanded.expansion_metadata["expanded"] is False
        assert "self-contained" in expanded.expansion_metadata.get("reason", "")

    def test_expand_batch(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test batch expansion of multiple chunks."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Create multiple retrieved chunks
        retrieved_chunks = [
            RetrievedChunk(
                chunk=sample_code_chunks["code_1"],
                rrf_score=0.6,
                request_reasoning="Need Client class",
                source_query="Client",
            ),
            RetrievedChunk(
                chunk=sample_code_chunks["code_2"],
                rrf_score=0.5,
                request_reasoning="Need GET method",
                source_query="GET",
            ),
        ]

        expanded_chunks = expander.expand_batch(retrieved_chunks)

        assert len(expanded_chunks) == 2
        assert all(isinstance(ec, ExpandedChunk) for ec in expanded_chunks)

    def test_expanded_chunk_total_tokens(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test that total_tokens includes expanded context."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Expand method chunk (should include parent + siblings)
        method_chunk = sample_code_chunks["code_2"]
        retrieved = RetrievedChunk(
            chunk=method_chunk,
            rrf_score=0.5,
            request_reasoning="Need GET method",
            source_query="GET",
        )

        expanded = expander.expand(retrieved)

        # Total tokens should be > base chunk tokens
        assert expanded.total_tokens >= method_chunk.token_count

        # If parent was found, tokens should include parent
        if expanded.parent_chunk:
            expected_min = method_chunk.token_count + expanded.parent_chunk.token_count
            assert expanded.total_tokens >= expected_min

    def test_expanded_chunk_get_full_context(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
        sample_code_chunks: dict[str, CodeChunk],
    ):
        """Test that get_full_context assembles all context."""
        chunk_loader = MockChunkLoader(sample_code_chunks)
        expander = ContextExpander(metadata_index, chunk_loader)

        # Expand method chunk
        method_chunk = sample_code_chunks["code_2"]
        retrieved = RetrievedChunk(
            chunk=method_chunk,
            rrf_score=0.5,
            request_reasoning="Need GET method",
            source_query="GET",
        )

        expanded = expander.expand(retrieved)
        full_context = expanded.get_full_context()

        # Should contain main chunk content
        assert method_chunk.content in full_context
        assert "# Main Chunk" in full_context

        # If parent found, should be in context
        if expanded.parent_chunk:
            assert "# Parent Context" in full_context
            assert expanded.parent_chunk.content in full_context


# =============================================================================
# Integration Tests (no LLM)
# =============================================================================


class TestRetrievalIntegration:
    """Integration tests for retrieval components (no LLM in the loop)."""

    def test_metadata_filter_to_folder_boost_flow(
        self,
        metadata_index: dict[str, list[ChunkMetadata]],
    ):
        """Test full flow: metadata filter → folder boost (simulated RRF)."""
        # 1. Create request with folders (soft filter)
        request = RetrievalRequest(
            query="transport implementation",
            source_types=["code"],
            folders=["httpx/_transports/"],
            reasoning="Need transport code",
        )

        # 2. Apply metadata filter (source_type only, folders ignored)
        filter_instance = MetadataFilter(metadata_index, request)
        candidate_ids = filter_instance.apply()

        # Should return ALL code chunks (folders not excluded)
        assert len(candidate_ids) == 5

        # 3. Simulate RRF scoring where a transport chunk starts lower than a non-transport chunk
        # This simulates a scenario where the transport chunk would lose without boost
        ranked_ids = [
            ("code_1", 0.50),  # Client class (httpx/_client.py) - NOT in transport folder
            ("code_4", 0.45),  # BaseTransport (httpx/_transports/base.py) - IN transport folder
            ("code_2", 0.40),  # Client.get (httpx/_client.py) - NOT in transport folder
            ("code_5", 0.35),  # HTTP11Transport (httpx/_transports/http11.py) - IN transport folder
            ("code_3", 0.30),  # Client.post (httpx/_client.py) - NOT in transport folder
        ]

        # 4. Build file path mapping
        chunk_file_paths = {}
        for source_type, chunks_meta in metadata_index.items():
            for meta in chunks_meta:
                chunk_file_paths[meta.id] = meta.file_path

        # 5. Apply folder boost
        folders = request.folders
        boost_factor = 1.3

        boosted_results = []
        for chunk_id, score in ranked_ids:
            file_path = chunk_file_paths.get(chunk_id, "")
            matches_folder = any(file_path.startswith(folder) for folder in folders)
            if matches_folder:
                boosted_score = score * boost_factor
            else:
                boosted_score = score
            boosted_results.append((chunk_id, boosted_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)

        # After boost:
        # code_4: 0.45 * 1.3 = 0.585 (boosted to #1)
        # code_1: 0.50 (no boost, now #2)
        # code_5: 0.35 * 1.3 = 0.455 (boosted to #3)
        # code_2: 0.40 (no boost, #4)
        # code_3: 0.30 (no boost, #5)

        # Transport chunk code_4 should be boosted to rank #1
        assert boosted_results[0][0] == "code_4"
        assert boosted_results[0][1] == pytest.approx(0.585, rel=1e-3)

        # Non-transport chunk code_1 should now be #2
        assert boosted_results[1][0] == "code_1"
        assert boosted_results[1][1] == 0.50

        # Transport chunk code_5 should be boosted to #3
        assert boosted_results[2][0] == "code_5"
        assert boosted_results[2][1] == pytest.approx(0.455, rel=1e-3)

    def test_retrieval_request_validation(self):
        """Test that RetrievalRequest validates inputs correctly."""
        # Valid request
        valid_request = RetrievalRequest(
            query="valid query text",
            source_types=["code"],
            reasoning="valid reasoning text",
        )
        assert valid_request.query == "valid query text"

        # Invalid: empty query
        with pytest.raises(ValueError):
            RetrievalRequest(
                query="    ",  # Empty after strip
                source_types=["code"],
                reasoning="valid reasoning",
            )

        # Invalid: empty source_types
        with pytest.raises(ValueError):
            RetrievalRequest(
                query="valid query",
                source_types=[],  # Empty
                reasoning="valid reasoning",
            )

        # Invalid: short reasoning
        with pytest.raises(ValueError):
            RetrievalRequest(
                query="valid query",
                source_types=["code"],
                reasoning="short",  # < 10 chars
            )
