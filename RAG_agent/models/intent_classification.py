"""
Intent classification models.

Defines Pydantic models for user intent classification in the RAG documentation assistant.
These models are used by the intent classification node to route requests appropriately.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

__all__ = ["IntentClassification", "ConversationContext", "ParsedIntentClassification"]


# Conversation context types
ConversationContext = Literal[
    "new_topic",              # User is asking about a completely new topic
    "continuing_topic",       # User is continuing discussion on the same topic
    "clarification",          # User is asking for clarification on previous answer
    "follow_up",              # User is asking a follow-up that needs additional RAG
]


# The 10 intent types for the documentation assistant
IntentType = Literal[
    "factual_lookup",         # Quick factual questions with direct answers
    "implementation_guide",   # Working code, how-to, tutorials
    "troubleshooting",        # Errors, bugs, unexpected behavior
    "conceptual_explanation", # Deep understanding of concepts
    "best_practices",         # Recommendations, patterns, optimization
    "comparison",             # Comparing features, frameworks, approaches
    "configuration_setup",    # Installation, environment, configuration
    "migration",              # Version upgrades, framework changes
    "capability_exploration", # Feature discovery, what's possible
    "api_reference",          # Specific API, method, class documentation
]


class IntentClassification(BaseModel):
    """Classification of user intent for routing and RAG optimization.

    This model captures the user's intent to help the RAG system:
    1. Optimize search queries (different intents need different retrieval strategies)
    2. Format responses appropriately (code vs explanations vs troubleshooting)
    3. Determine if clarification is needed before processing
    4. Handle conversation context (new topic vs continuation vs clarification)
    """

    intent_type: IntentType = Field(
        ...,
        description="Primary intent category for RAG optimization and response formatting",
    )
    framework: Optional[List[Literal["langchain", "langgraph", "langsmith", "general"]]] = Field(
        default=None,
        description=(
            "Framework context(s) for document filtering. Can be a list for multi-framework queries. "
            "Examples: ['langchain'], ['langgraph'], ['langchain', 'langgraph'] for both"
        ),
    )
    language: Optional[List[Literal["python", "javascript"]]] = Field(
        default=None,
        description=(
            "Programming language context(s) for code examples. Can be a list for multi-language queries. "
            "Examples: ['python'], ['javascript'], ['python', 'javascript'] for both. "
            "None if not specified (defaults to Python in responses)."
        ),
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Specific topics for search expansion (1-5 terms)",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Key technical terms from user input, ordered by importance (max 10)",
    )
    conversation_context: ConversationContext = Field(
        default="new_topic",
        description=(
            "Conversation context: new_topic (fresh query), continuing_topic (same topic continuation), "
            "clarification (asking about previous answer), follow_up (needs additional RAG)"
        ),
    )
    requires_rag: bool = Field(
        default=True,
        description=(
            "Whether RAG retrieval is needed. Default True for documentation assistant. "
            "False for clarifications that don't need new docs or universal CS/AI concepts."
        ),
    )

    @model_validator(mode="after")
    def validate_requires_rag(self) -> "IntentClassification":
        """Validate that requires_rag=False only when conditions are met.

        RAG can only be skipped when:
        1. Conversation context is "clarification" (asking about previous answer)
        2. Framework is "general" or None (not framework-specific) for universal concepts
        3. Question is about universal concepts (not implementation-specific)
        """
        if not self.requires_rag:
            # Allow skipping RAG for clarifications on previous answers
            if self.conversation_context == "clarification":
                return self

            # For non-clarification queries, require general or unspecified framework
            if self.framework:
                # Check if any framework is not "general"
                specific_frameworks = [f for f in self.framework if f != "general"]
                if specific_frameworks:
                    raise ValueError(
                        f"requires_rag=False not allowed for framework-specific questions "
                        f"(framework={self.framework}). Framework-specific questions need RAG."
                    )

            # Implementation-focused intents always need RAG (unless clarification)
            implementation_intents = {
                "implementation_guide",
                "troubleshooting",
                "api_reference",
                "configuration_setup",
                "migration",
            }
            if self.intent_type in implementation_intents:
                raise ValueError(
                    f"requires_rag=False not allowed for {self.intent_type}. "
                    "Implementation-focused intents require RAG."
                )

        return self


class ParsedIntentClassification(BaseModel):
    """
    Combined model for request parsing AND intent classification (merged into one LLM call).

    This model combines both ParsedRequest fields and IntentClassification fields
    to reduce latency by getting everything in a single LLM invocation.
    """

    # =========================================================================
    # PARSING FIELDS (from ParsedRequest)
    # =========================================================================

    cleaned_request: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Cleaned and normalized request(s) with conversational fluff removed"
    )

    code_snippets: Optional[List[str]] = Field(
        default=None,
        description="Extracted code blocks (if any), preserved exactly as written"
    )

    data: Optional[dict] = Field(
        default=None,
        description="Extracted structured data (errors, configurations, etc.)"
    )

    query_type: Literal["question", "instruction", "troubleshooting"] = Field(
        ...,
        description="Type of user query"
    )

    security_analysis: Optional[dict] = Field(
        default=None,
        description="Security analysis results (suspicious patterns, risk level, reasoning)"
    )

    # =========================================================================
    # CLASSIFICATION FIELDS (from IntentClassification)
    # =========================================================================

    intent_type: IntentType = Field(
        ...,
        description="Primary user intent category"
    )

    framework: Optional[List[Literal["langchain", "langgraph", "langsmith", "general"]]] = Field(
        default=None,
        description="Target framework(s)"
    )

    language: Optional[Literal["python", "javascript", "typescript"]] = Field(
        default=None,
        description="Programming language"
    )

    topics: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Key topics from knowledge base taxonomy"
    )

    keywords: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Search keywords ordered by importance"
    )

    conversation_context: ConversationContext = Field(
        ...,
        description="Conversation context type"
    )

    requires_rag: bool = Field(
        ...,
        description="Whether RAG retrieval is needed"
    )

    @model_validator(mode="after")
    def validate_rag_requirement(self) -> "ParsedIntentClassification":
        """Validate that requires_rag=False is only used appropriately."""
        if not self.requires_rag:
            # Clarifications are allowed to skip RAG
            if self.conversation_context == "clarification":
                return self

            # For non-clarification queries, require general or unspecified framework
            if self.framework:
                specific_frameworks = [f for f in self.framework if f != "general"]
                if specific_frameworks:
                    raise ValueError(
                        f"requires_rag=False not allowed for framework-specific questions "
                        f"(framework={self.framework})"
                    )

            # Implementation-focused intents always need RAG
            implementation_intents = {
                "implementation_guide",
                "troubleshooting",
                "api_reference",
                "configuration_setup",
                "migration",
            }
            if self.intent_type in implementation_intents:
                raise ValueError(
                    f"requires_rag=False not allowed for {self.intent_type}"
                )

        return self


