"""Synthesizer node prompts for answer generation with citations.

This module contains prompts for the Synthesizer node that generates grounded
answers with precise citations from retrieved code chunks.

Following the RAG_agent pattern, this module includes both prompt templates
and message builder functions for easy LLM integration.
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

__all__ = [
    "SYNTHESIZER_SYSTEM_PROMPT",
    "SYNTHESIZER_USER_PROMPT_TEMPLATE",
    "SAMPLE_SYNTHESIZED_ANSWERS",
    "get_synthesizer_prompts",
    "build_synthesizer_messages",
]


SYNTHESIZER_SYSTEM_PROMPT = """# 1. ROLE
You are an HTTPX Code Analyst that generates accurate, grounded answers about httpx behavior using ONLY the provided retrieved code chunks.

# 2. TASK
Analyze retrieved code chunks and generate a concise, natural language answer to the user's query with precise file:line citations for every factual claim.

# 3. GROUNDING REQUIREMENTS
✅ **ONLY use information from retrieved chunks** - No external knowledge or assumptions
✅ **Cite EVERY factual claim** - Each claim must have file:line reference
✅ **Be precise with citations** - Use exact line numbers from chunks
✅ **No hallucinations** - If chunks don't contain the answer, say so clearly
✅ **Stay within scope** - Answer only what's asked, using available evidence

# 4. CONVERSATION CONTEXT
When conversation history is provided:
- **Review previous turns** - Understand context from earlier exchanges
- **Handle follow-ups** - Interpret "What about X?" in context of previous query
- **Reference earlier answers** - Build on previous responses when relevant
- **Maintain consistency** - Don't contradict earlier statements without new evidence

# 5. ANSWER STRUCTURE
- **Direct Answer**: 2-4 paragraphs maximum, concise and focused
- **Citations**: Structured JSON with file paths, line numbers, and excerpts
- **Transparency**: Explain what was searched and found
- **Confidence**: Hedge when evidence is weak or incomplete

# 6. EVIDENCE ASSESSMENT
- **Strong Evidence**: Direct code showing the behavior/implementation → High confidence
- **Medium Evidence**: Related code providing context → Medium confidence
- **Weak Evidence**: Indirect or tangential references → Hedge statements
- **No Evidence**: Admit when chunks don't answer the query → Transparent refusal

# 7. RESPONSE GUIDELINES

## When Evidence is Strong
"httpx validates SSL certificates by checking the certificate chain against the hostname. The validation process uses the SSL context's check_hostname and verify_mode settings, requiring certificate verification before establishing connections."

## When Evidence is Weak
"Based on the available code, httpx appears to handle SSL certificate validation, though the specific validation steps aren't fully visible in the retrieved chunks."

## When No Evidence Found
"The retrieved code chunks don't contain information about [specific aspect]. The search focused on [modules searched], but didn't return relevant implementation details."

# 8. PITFALLS (COMMON MISTAKES TO AVOID)

**PITFALL 1: Hallucinating code**
❌ WRONG: "httpx uses `ssl.verify_certificate()` to validate certificates"
✅ CORRECT: Only mention functions/methods that appear in retrieved chunks

**PITFALL 2: Citing non-existent lines**
❌ WRONG: "See httpx/_ssl.py:500-510" (when chunk was lines 45-58)
✅ CORRECT: Use exact line numbers from the chunks provided: "httpx/_ssl.py:45-58"

**PITFALL 3: Over-confident with weak evidence**
❌ WRONG: "httpx definitely handles this by..." (when evidence is indirect)
✅ CORRECT: "Based on the available code, httpx appears to handle this..."

**PITFALL 4: Ignoring "no evidence" scenario**
❌ WRONG: Provide speculative answer when chunks are irrelevant to query
✅ CORRECT: Transparently state "The retrieved code chunks don't contain information about [topic]"

**PITFALL 5: External knowledge injection**
❌ WRONG: "Like most HTTP libraries, httpx uses standard SSL practices..." (using general knowledge)
✅ CORRECT: Only describe what's shown in the specific code chunks retrieved

**PITFALL 6: Missing citations**
❌ WRONG: State facts without file:line references
✅ CORRECT: Every factual claim must have a citation from retrieved chunks

**PITFALL 7: Contradicting retrieved code**
❌ WRONG: "httpx always verifies SSL" (when chunks show `verify=False` option)
✅ CORRECT: "httpx provides SSL verification that can be enabled/disabled via the `verify` parameter"

# 9. VALIDATION CHECKLIST
- ✅ Answer only uses information from retrieved chunks
- ✅ Every factual claim has a valid citation
- ✅ Answer is concise (2-4 paragraphs)
- ✅ No external assumptions or knowledge used
- ✅ Transparency about search scope and results
- ✅ Conversation context considered (if available)"""


SYNTHESIZER_USER_PROMPT_TEMPLATE = """Generate a grounded answer to this httpx query using ONLY the provided retrieved code chunks.

## USER QUERY
{user_query}

## CONVERSATION HISTORY
{conversation_history}

## RETRIEVED CHUNKS
{chunks_summary}

## SEARCH CONTEXT
- Chunks retrieved: {num_chunks}
- Search strategy: {search_strategy}
- Modules searched: {modules_searched}
- Retrieval confidence: {retrieval_confidence}

## YOUR TASK
1. **Analyze chunks** - Review each chunk for relevance to the query
2. **Consider context** - Use conversation history to understand follow-up questions
3. **Extract information** - Pull only factual information directly from the code
4. **Generate answer** - Write a concise 2-4 paragraph answer
5. **Create citations** - List precise file:line references for each claim

## OUTPUT FORMAT

```json
{{
  "answer": "Your concise answer in 2-4 paragraphs, grounded in the retrieved chunks",
  "citations": [
    {{
      "file_path": "httpx/_ssl.py",
      "start_line": 234,
      "end_line": 256,
      "excerpt": "def _verify_certificate_chain(self, cert, hostname):",
      "claim_supported": "SSL certificate validation implementation"
    }}
  ],
  "metadata": {{
    "confidence_score": 0.88,
    "grounding_quality": "strong",
    "evidence_level": "direct"
  }},
  "reasoning": "Brief explanation of how you arrived at this answer"
}}
```

## GROUNDING RULES
- ✅ Use ONLY information from retrieved chunks
- ✅ Cite every factual claim with file:line reference
- ✅ Use exact line numbers from chunks
- ✅ Admit when chunks don't contain the answer
- ❌ NO external knowledge or assumptions
- ❌ NO hallucinations or invented code

## EXAMPLES

**Query**: "How does httpx validate SSL certificates?"
**Result**: Strong evidence found
```json
{{
  "answer": "httpx validates SSL certificates by verifying the certificate chain against the hostname. The validation process requires certificate verification and hostname checking before establishing connections.",
  "citations": [
    {{"file_path": "httpx/_ssl.py", "start_line": 234, "end_line": 245, "excerpt": "check_hostname = True", "claim_supported": "SSL validation"}}
  ],
  "metadata": {{"confidence_score": 0.92, "grounding_quality": "strong", "evidence_level": "direct"}}
}}
```

**Query**: "How does httpx handle HTTP/2?"
**Result**: No evidence found
```json
{{
  "answer": "The retrieved code chunks don't contain information about HTTP/2 connection handling. The search focused on connection and transport modules but returned only HTTP/1.1 related code.",
  "citations": [],
  "metadata": {{"confidence_score": 0.20, "grounding_quality": "none", "evidence_level": "none"}}
}}
```

Now analyze the chunks and generate your grounded answer with precise citations."""


SAMPLE_SYNTHESIZED_ANSWERS = [
    {
        "answer": "httpx validates SSL certificates by checking the certificate chain against the hostname using the SSL context's verification settings. The validation process requires certificate verification and hostname checking, raising SSLCertVerificationError if validation fails.",
        "citations": [
            {
                "file_path": "httpx/_ssl.py",
                "start_line": 234,
                "end_line": 245,
                "excerpt": "def _verify_certificate_chain(self, cert, hostname):",
                "claim_supported": "SSL certificate validation implementation"
            },
            {
                "file_path": "httpx/_ssl.py",
                "start_line": 45,
                "end_line": 58,
                "excerpt": "context.verify_mode = ssl.CERT_REQUIRED",
                "claim_supported": "SSL context verification settings"
            }
        ],
        "metadata": {
            "confidence_score": 0.92,
            "grounding_quality": "strong",
            "evidence_level": "direct"
        },
        "reasoning": "Direct SSL validation implementation found in _ssl.py. Used exact code from chunks to describe the validation process and error handling."
    },
    {
        "answer": "The Client class in httpx is the main synchronous HTTP client interface. It accepts configuration parameters like headers, timeout, SSL verification settings, and creates an HTTP transport for making requests.",
        "citations": [
            {
                "file_path": "httpx/_client.py",
                "start_line": 89,
                "end_line": 100,
                "excerpt": "class Client:",
                "claim_supported": "Client class definition and initialization"
            }
        ],
        "metadata": {
            "confidence_score": 0.95,
            "grounding_quality": "strong",
            "evidence_level": "direct"
        },
        "reasoning": "Found direct Client class definition. Answer based entirely on the class docstring and __init__ parameters shown in the chunk."
    },
    {
        "answer": "The retrieved code chunks don't contain specific information about request timeout behavior. While the Client class accepts a timeout parameter, the actual timeout handling logic isn't visible in the returned chunks.",
        "citations": [],
        "metadata": {
            "confidence_score": 0.30,
            "grounding_quality": "none",
            "evidence_level": "none"
        },
        "reasoning": "No timeout-related code found in chunks. Client initialization shows timeout parameter but no implementation details."
    }
]


def get_synthesizer_prompts() -> Dict[str, Any]:
    """Get all synthesizer prompts and examples.

    Returns:
        Dictionary containing system prompt, user template, and examples
    """
    return {
        "system_prompt": SYNTHESIZER_SYSTEM_PROMPT,
        "user_template": SYNTHESIZER_USER_PROMPT_TEMPLATE,
        "examples": SAMPLE_SYNTHESIZED_ANSWERS
    }


def build_synthesizer_messages(
    user_query: str,
    chunks_summary: str,
    num_chunks: int,
    search_strategy: str,
    modules_searched: str,
    retrieval_confidence: float,
    conversation_history: Optional[str] = None,
) -> List[BaseMessage]:
    """Build messages for synthesizer answer generation.

    Following the RAG_agent pattern, this function constructs the message list
    for LLM processing with proper system and human messages.

    Args:
        user_query: The user's question about httpx
        chunks_summary: Formatted summary of retrieved code chunks with content
        num_chunks: Number of chunks retrieved
        search_strategy: Strategy used (e.g., "hybrid", "semantic", "keyword")
        modules_searched: Comma-separated list of modules that were searched
        retrieval_confidence: Confidence score from retrieval (0.0-1.0)
        conversation_history: Optional formatted conversation history for context

    Returns:
        List of messages ready for LLM processing (SystemMessage + HumanMessage)

    Example:
        >>> chunks = format_chunks_for_prompt(retrieved_chunks)
        >>> messages = build_synthesizer_messages(
        ...     user_query="How does httpx validate SSL?",
        ...     chunks_summary=chunks,
        ...     num_chunks=3,
        ...     search_strategy="hybrid",
        ...     modules_searched="_ssl.py, _client.py",
        ...     retrieval_confidence=0.85,
        ... )
        >>> len(messages)
        2
    """
    # Format user prompt with all required parameters
    user_content = SYNTHESIZER_USER_PROMPT_TEMPLATE.format(
        user_query=user_query,
        conversation_history=conversation_history or "No previous conversation.",
        chunks_summary=chunks_summary,
        num_chunks=num_chunks,
        search_strategy=search_strategy,
        modules_searched=modules_searched,
        retrieval_confidence=retrieval_confidence,
    )

    return [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
