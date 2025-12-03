"""Pydantic models for document structure."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Link(BaseModel):
    """Represents a link extracted from markdown."""

    id: str = Field(description="Unique identifier for LLM search (normalized text)")
    url: str = Field(description="Link URL")


class CodeBlock(BaseModel):
    """Represents a code block extracted from markdown."""

    content: str = Field(description="The actual code content")
    language: Optional[str] = Field(
        default=None, description="Programming language or framework (e.g., python, bash, mermaid, javascript)"
    )


class TextBlock(BaseModel):
    """Represents a text content block."""

    type: Literal["text"] = "text"
    content: str = Field(description="Text content")


class CodeContentBlock(BaseModel):
    """Represents a code block in sequential content."""

    type: Literal["code"] = "code"
    content: str = Field(description="The actual code content")
    header: Optional[str] = Field(
        default=None, description="The text/header that precedes this code block, providing context"
    )


class TableContentBlock(BaseModel):
    """Represents a table block in sequential content."""

    type: Literal["table"] = "table"
    content: str = Field(description="Table content in CSV format")


ContentBlock = Union[TextBlock, CodeContentBlock, TableContentBlock]


class DocumentSection(BaseModel):
    """Represents a section in a markdown document with hierarchical structure."""

    title: str = Field(description="Section heading text")
    level: int = Field(description="Heading level (1-6, where 1 is H1)")
    links: List[Link] = Field(
        default_factory=list, description="Extracted links with identifiers for LLM search"
    )
    content_blocks: Optional[List[ContentBlock]] = Field(
        default=None, description="Sequential content blocks preserving order of text and code for LLM consumption"
    )
    children: List["DocumentSection"] = Field(
        default_factory=list, description="Child sections as a list"
    )

    class Config:
        """Pydantic config."""

        json_encoders = {
            # Ensure proper JSON serialization
        }


# Update forward reference
DocumentSection.model_rebuild()

