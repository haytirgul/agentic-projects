"""Conversation memory models for storing query-response history.

This module defines the structure for storing conversation turns in memory.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from models.intent_classification import ParsedIntentClassification

__all__ = ["ConversationTurn", "ConversationMemory"]


class ConversationTurn(BaseModel):
    """A single turn in the conversation (user query + assistant response).

    This stores the complete context of one interaction cycle.
    """

    # Original user input
    user_query: str = Field(description="Original user query")

    # Parsed/refined query data
    cleaned_request: str = Field(description="Cleaned and parsed user request")
    intent_classification: Optional[ParsedIntentClassification] = Field(
        default=None,
        description="Combined parsing + intent classification result"
    )
    code_snippets: List[str] = Field(
        default_factory=list,
        description="Code snippets extracted from user input"
    )

    # Assistant response
    assistant_response: str = Field(description="Final response from the assistant")

    # Metadata
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of this turn"
    )
    retrieved_doc_paths: List[str] = Field(
        default_factory=list,
        description="File paths of documents used in this response"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationMemory(BaseModel):
    """Complete conversation history for the session.

    Stores all turns in chronological order.
    """

    turns: List[ConversationTurn] = Field(
        default_factory=list,
        description="List of conversation turns in chronological order"
    )

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new conversation turn to history.

        Args:
            turn: ConversationTurn to add
        """
        self.turns.append(turn)

    def get_last_n_turns(self, n: int = 3) -> List[ConversationTurn]:
        """Get the last N conversation turns.

        Args:
            n: Number of recent turns to retrieve

        Returns:
            List of the most recent N turns
        """
        return self.turns[-n:] if self.turns else []

    def get_context_summary(self, n: int = 3) -> str:
        """Get a text summary of recent conversation context.

        Args:
            n: Number of recent turns to include

        Returns:
            Formatted string summarizing recent conversation
        """
        recent_turns = self.get_last_n_turns(n)
        if not recent_turns:
            return "No previous conversation."

        summary_parts = ["## Recent Conversation Context\n"]
        for i, turn in enumerate(recent_turns, 1):
            summary_parts.append(f"### Turn {i}")
            summary_parts.append(f"**User**: {turn.user_query}")
            if turn.intent_classification:
                summary_parts.append(
                    f"**Intent**: {turn.intent_classification.intent_type} "
                    f"({turn.intent_classification.framework or 'general'})"
                )
            summary_parts.append(f"**Assistant**: {turn.assistant_response[:200]}...")
            summary_parts.append("")

        return "\n".join(summary_parts)

    @property
    def current_topic(self) -> str:
        """Get the current conversation topic from the most recent turn.

        Returns:
            Topic string (e.g., "persistence", "agents", etc.) from the latest intent classification,
            or empty string if no conversation or no classified intent.
        """
        if not self.turns:
            return ""

        latest_turn = self.turns[-1]
        if latest_turn.intent_classification and latest_turn.intent_classification.topics:
            # Return the primary topic (first one)
            return latest_turn.intent_classification.topics[0]

        return ""

    def to_json_serializable(self) -> dict:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "turns": [turn.dict() for turn in self.turns]
        }
