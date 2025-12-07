"""Conversation memory models for storing query-response history.

This module defines the structure for storing conversation turns in memory
for multi-turn code analysis conversations.
"""

from datetime import datetime

from pydantic import BaseModel, Field

__all__ = ["ConversationTurn", "ConversationMemory"]


class ConversationTurn(BaseModel):
    """Pydantic model for a single conversation turn.

    Represents one query-response pair in the conversation history.
    """

    user_query: str = Field(..., description="User's question about the codebase")
    assistant_response: str = Field(..., description="Agent's answer with citations")
    citations: list[str] = Field(
        default_factory=list,
        description="File:line citations used in response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Turn timestamp"
    )
    chunks_retrieved: int = Field(
        0,
        ge=0,
        description="Number of code chunks retrieved for this turn"
    )
    retrieval_queries: list[str] = Field(
        default_factory=list,
        description="Queries used to retrieve code chunks"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConversationMemory(BaseModel):
    """Pydantic model for conversation memory management.

    Maintains conversation history with configurable window size for
    multi-turn code analysis sessions.
    """

    turns: list[ConversationTurn] = Field(
        default_factory=list,
        description="Conversation history in chronological order"
    )
    max_turns: int = Field(
        5,
        ge=1,
        description="Maximum turns to keep in memory"
    )
    session_id: str = Field(..., description="Unique session identifier")
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session start time"
    )

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn and maintain max window size.

        Args:
            turn: ConversationTurn to add to history

        Note:
            Automatically removes oldest turn when max_turns is exceeded.
        """
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_recent_context(self, n: int = 3) -> str:
        """Get formatted recent conversation context.

        Args:
            n: Number of recent turns to include (default: 3)

        Returns:
            Formatted string with recent conversation history
        """
        recent = self.turns[-n:] if len(self.turns) > n else self.turns
        if not recent:
            return ""

        context_parts = []
        for i, turn in enumerate(recent, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"User: {turn.user_query}")
            # Truncate long responses for context
            response_preview = turn.assistant_response[:300]
            if len(turn.assistant_response) > 300:
                response_preview += "..."
            context_parts.append(f"Assistant: {response_preview}")
            if turn.citations:
                context_parts.append(f"Citations: {', '.join(turn.citations[:3])}")
            context_parts.append("")  # Blank line separator

        return "\n".join(context_parts)

    def get_last_n_turns(self, n: int = 3) -> list[ConversationTurn]:
        """Get the last N conversation turns.

        Args:
            n: Number of recent turns to retrieve

        Returns:
            list of the most recent N turns
        """
        return self.turns[-n:] if self.turns else []

    @property
    def turn_count(self) -> int:
        """Get total number of turns in memory.

        Returns:
            Number of conversation turns stored
        """
        return len(self.turns)

    @property
    def last_query(self) -> str:
        """Get the most recent user query.

        Returns:
            Last user query or empty string if no history
        """
        if not self.turns:
            return ""
        return self.turns[-1].user_query

    def to_json_serializable(self) -> dict:
        """Convert to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "turn_count": self.turn_count,
            "turns": [turn.dict() for turn in self.turns]
        }

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
