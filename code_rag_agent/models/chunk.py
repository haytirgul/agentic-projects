"""Pydantic models for chunk data structures.

This module defines the unified chunk schema for all document types (code, markdown, text).
All chunks share a common base structure with type-specific optional fields.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BaseChunk(BaseModel):
    """Base chunk model with fields common to all chunk types.

    This model provides a unified schema for code, markdown, and text chunks,
    enabling consistent metadata filtering and retrieval across all document types.
    """

    # Core identification
    id: str = Field(..., description="Unique chunk identifier (MD5 hash)")
    source_type: Literal["code", "markdown", "text"] = Field(
        ...,
        description="High-level document category for easy filtering"
    )
    chunk_type: str = Field(
        ...,
        description="Specific chunk subtype (e.g., 'function', 'markdown_section')"
    )

    # Content
    content: str = Field(..., description="The actual chunk text content")
    token_count: int = Field(..., ge=0, description="Estimated token count for context window management")

    # File location
    file_path: str = Field(..., description="Relative path from repository root")
    filename: str = Field(..., description="Base filename")
    start_line: int = Field(..., ge=0, description="Starting line number in original file")
    end_line: int = Field(..., ge=0, description="Ending line number in original file")

    # Context
    name: str = Field(..., description="Chunk name (function name, header title, etc.)")
    full_name: str = Field(..., description="Fully qualified name with parent context")
    parent_context: str = Field(default="", description="Parent context (class name, file name, etc.)")

    # Type-specific fields (optional, populated based on source_type)
    docstring: str | None = Field(default=None, description="Docstring (code only)")
    imports: list[str] = Field(default_factory=list, description="Import statements (code only)")
    headers: dict[str, str] = Field(default_factory=dict, description="Header hierarchy (markdown only)")
    heading_level: int | None = Field(default=None, ge=1, le=6, description="Heading level (markdown only)")
    is_code: bool | None = Field(default=None, description="Whether content is code (markdown only)")
    file_extension: str | None = Field(default=None, description="File extension (text only)")

    @field_validator('chunk_type')
    @classmethod
    def validate_chunk_type(cls, v: str, info) -> str:
        """Validate chunk_type matches source_type."""
        source_type = info.data.get('source_type')

        valid_types = {
            'code': ['function', 'class'],
            'markdown': ['markdown_section', 'markdown_section_chunk'],
            'text': ['text_file', 'text_file_chunk']
        }

        if source_type and source_type in valid_types:
            if v not in valid_types[source_type]:
                raise ValueError(
                    f"chunk_type '{v}' not valid for source_type '{source_type}'. "
                    f"Valid types: {valid_types[source_type]}"
                )

        return v

    @field_validator('end_line')
    @classmethod
    def validate_line_range(cls, v: int, info) -> int:
        """Ensure end_line >= start_line."""
        start_line = info.data.get('start_line', 0)
        if v < start_line:
            raise ValueError(f"end_line ({v}) must be >= start_line ({start_line})")
        return v


class CodeChunk(BaseChunk):
    """Code-specific chunk model for Python functions and classes.

    Inherits all base fields and enforces code-specific constraints.
    """
    source_type: Literal["code"] = "code"
    chunk_type: Literal["function", "class"]

    # Required for code chunks
    docstring: str | None = None
    imports: list[str] = Field(default_factory=list)

    # Not applicable for code
    headers: dict[str, str] = Field(default_factory=dict, exclude=True)
    heading_level: int | None = Field(default=None, exclude=True)
    is_code: bool | None = Field(default=None, exclude=True)
    file_extension: str | None = Field(default=None, exclude=True)


class MarkdownChunk(BaseChunk):
    """Markdown-specific chunk model for documentation sections.

    Inherits all base fields and enforces markdown-specific constraints.
    """
    source_type: Literal["markdown"] = "markdown"
    chunk_type: Literal["markdown_section", "markdown_section_chunk"]

    # Required for markdown chunks
    headers: dict[str, str] = Field(default_factory=dict)
    heading_level: int | None = Field(default=1, ge=1, le=6)
    is_code: bool = False

    # Not applicable for markdown
    docstring: str | None = Field(default=None, exclude=True)
    imports: list[str] = Field(default_factory=list, exclude=True)
    file_extension: str | None = Field(default=None, exclude=True)


class TextChunk(BaseChunk):
    """Text file chunk model for configuration and data files.

    Inherits all base fields and enforces text-specific constraints.
    """
    source_type: Literal["text"] = "text"
    chunk_type: Literal["text_file", "text_file_chunk"]

    # Required for text chunks
    file_extension: str

    # Not applicable for text
    docstring: str | None = Field(default=None, exclude=True)
    imports: list[str] = Field(default_factory=list, exclude=True)
    headers: dict[str, str] = Field(default_factory=dict, exclude=True)
    heading_level: int | None = Field(default=None, exclude=True)
    is_code: bool | None = Field(default=None, exclude=True)


# Type alias for any chunk type
Chunk = CodeChunk | MarkdownChunk | TextChunk


def create_chunk_from_dict(data: dict) -> Chunk:
    """Factory function to create appropriate chunk model from dictionary.

    Args:
        data: Dictionary containing chunk data with 'source_type' field

    Returns:
        Appropriate chunk model instance (CodeChunk, MarkdownChunk, or TextChunk)

    Raises:
        ValueError: If source_type is invalid or missing
    """
    source_type = data.get('source_type')

    if source_type == 'code':
        return CodeChunk(**data)
    elif source_type == 'markdown':
        return MarkdownChunk(**data)
    elif source_type == 'text':
        return TextChunk(**data)
    else:
        raise ValueError(f"Invalid source_type: {source_type}. Must be 'code', 'markdown', or 'text'")
