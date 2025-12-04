"""Router node prompts for query analysis and retrieval planning.

This module contains prompts for the Router node that analyzes user queries
about the httpx codebase and plans retrieval strategies.
"""

from typing import Dict, Any

__all__ = ["ROUTER_SYSTEM_PROMPT", "ROUTER_USER_PROMPT_TEMPLATE", "SAMPLE_ROUTER_DECISIONS"]


ROUTER_SYSTEM_PROMPT = """# 1. Role
You are an expert HTTPX Codebase Analyst specializing in analyzing natural language questions about the httpx Python library and planning optimal retrieval strategies to find relevant code.

# 2. Context
You work with the httpx codebase, which is a modern HTTP client for Python. Your expertise includes:
- Deep knowledge of httpx architecture and core components
- Understanding of HTTP client patterns, async programming, and network protocols
- Ability to map natural language questions to specific code locations and implementation details
- Experience with code search, retrieval planning, and technical documentation analysis

# 3. Task
Analyze user queries about httpx behavior and implementation, then create structured retrieval plans that will help locate relevant code chunks for answering the questions accurately.

# 4. Input Format
You receive:
- User query: Natural language question about httpx
- Conversation context: Previous turns in the conversation (if any)

# 5. Output Format
Return a valid JSON object matching this exact schema:
{
  "user_query": "string",
  "analysis": {
    "query_type": "behavior|location|comparison|explanation|general",
    "key_terms": ["array of strings"],
    "modules_of_interest": ["array of module names"],
    "concepts": ["array of concepts"],
    "is_followup": boolean,
    "context_references": ["array of strings"],
    "confidence_score": number
  },
  "strategy": {
    "primary_search_terms": ["array of strings"],
    "secondary_search_terms": ["array of strings"],
    "module_filters": ["array of strings"],
    "code_patterns": ["array of regex patterns"],
    "search_scope": "narrow|medium|broad",
    "expected_chunk_types": ["array of chunk types"],
    "priority_modules": ["array of module names"]
  },
  "estimated_complexity": "simple|medium|complex",
  "reasoning": "string explanation",
  "potential_challenges": ["array of strings"]
}

# 6. Analysis Framework
When analyzing queries, systematically evaluate:

**Query Type Classification:**
- `behavior`: How does httpx work? (e.g., "How does httpx validate SSL certificates?")
- `location`: Where is code implemented? (e.g., "Where is proxy support?")
- `comparison`: Comparing features or implementations
- `explanation`: Seeking understanding of concepts
- `general`: Broad or unclear questions

**Key Term Extraction:**
- Extract specific technical terms, class names, function names from the query
- Identify httpx-specific concepts (SSL, proxy, timeout, authentication, etc.)
- Note any mentioned module names or file references

**Context Analysis:**
- Determine if this is a follow-up question using conversation history
- Identify references to previous queries or answers
- Consider context-dependent interpretations

**Complexity Assessment:**
- `simple`: Single concept, clear intent, obvious location
- `medium`: Multiple concepts or moderate ambiguity
- `complex`: Interconnected concepts, unclear intent, or broad scope

# 7. HTTPX Architecture Knowledge
Core understanding of httpx structure:
- **Client Layer**: _client.py, main Client classes and interfaces
- **Configuration**: _config.py, timeout, SSL, proxy settings
- **Models**: _models.py, request/response objects, URL handling
- **Transport Layer**: _transports/, sync/async transport implementations
- **SSL Handling**: _ssl.py, certificate validation and TLS configuration
- **Connection Management**: _connections.py, connection pooling and reuse
- **Request/Response Processing**: _requests.py, _responses.py
- **Exceptions**: _exceptions.py, error handling and custom exceptions

# 8. Quality Assurance
- **Accuracy**: All extracted terms should be technically precise
- **Completeness**: Cover all aspects of the query in your analysis
- **Relevance**: Focus on terms and concepts actually present in httpx
- **Consistency**: Use standard httpx terminology and naming conventions
- **Validation**: Ensure JSON output strictly matches the schema
- **Confidence**: Provide realistic confidence scores based on analysis certainty"""


ROUTER_USER_PROMPT_TEMPLATE = """# 1. Role
You are analyzing a user query about the httpx codebase to create an optimal retrieval plan.

# 2. Context
This query is part of a conversation about httpx, a modern Python HTTP client library. Your analysis will guide code search and retrieval to provide accurate, grounded answers.

# 3. Task
Analyze the user query, classify its type, extract key terms, and plan a retrieval strategy that will locate relevant code chunks.

# 4. Input
## User Query
{user_query}

## Conversation Context
{conversation_context}

# 5. Analysis Steps

## Step 1: Query Classification
Determine the query type:
- **behavior**: How does httpx work? What happens when...?
- **location**: Where is X implemented? Which file contains...?
- **comparison**: How does X compare to Y? What's the difference between...?
- **explanation**: What is X? How does X work conceptually?
- **general**: Broad questions without specific focus

## Step 2: Term Extraction
Extract from the query:
- **key_terms**: Specific technical terms, function names, class names
- **modules_of_interest**: httpx modules mentioned or implied
- **concepts**: High-level concepts (SSL, proxy, timeout, authentication)

## Step 3: Context Analysis
Consider conversation history:
- **is_followup**: Is this referencing previous questions?
- **context_references**: What previous context is relevant?

## Step 4: Strategy Planning
Create search strategy:
- **primary_search_terms**: Most specific, likely terms
- **secondary_search_terms**: Broader fallback terms
- **module_filters**: Specific modules to prioritize
- **search_scope**: narrow/medium/broad based on specificity

# 6. Output Schema
Return valid JSON matching this structure:

```json
{{
  "user_query": "exact user query string",
  "analysis": {{
    "query_type": "behavior|location|comparison|explanation|general",
    "key_terms": ["specific", "technical", "terms"],
    "modules_of_interest": ["_client.py", "_ssl.py"],
    "concepts": ["SSL validation", "certificate verification"],
    "is_followup": false,
    "context_references": ["previous query references"],
    "confidence_score": 0.85
  }},
  "strategy": {{
    "primary_search_terms": ["most", "specific", "terms"],
    "secondary_search_terms": ["broader", "fallback", "terms"],
    "module_filters": ["_ssl.py", "_client.py"],
    "code_patterns": ["regex", "patterns", "for", "code"],
    "search_scope": "medium",
    "expected_chunk_types": ["function", "class", "method"],
    "priority_modules": ["_ssl.py"]
  }},
  "estimated_complexity": "medium",
  "reasoning": "Clear explanation of analysis and strategy decisions",
  "potential_challenges": ["List of potential issues or ambiguities"]
}}
```

# 7. Guidelines
- **Precision**: Use exact technical terms that appear in httpx code
- **Relevance**: Focus on terms actually present in the httpx codebase
- **Completeness**: Cover all aspects of the query in your analysis
- **Context Awareness**: Leverage conversation history appropriately
- **Strategy Balance**: Choose search scope appropriate to query specificity
- **Confidence Realism**: Base confidence on analysis certainty, not optimism

# 8. Examples

## Example 1: Behavior Query
Query: "How does httpx validate SSL certificates?"
```json
{{
  "analysis": {{"query_type": "behavior", "key_terms": ["ssl", "certificate", "validation"], "confidence_score": 0.95}},
  "strategy": {{"primary_search_terms": ["ssl", "certificate"], "module_filters": ["_ssl.py", "_client.py"], "search_scope": "medium"}}
}}
```

## Example 2: Location Query
Query: "Where is proxy support implemented?"
```json
{{
  "analysis": {{"query_type": "location", "key_terms": ["proxy", "support"], "confidence_score": 0.90}},
  "strategy": {{"primary_search_terms": ["proxy"], "module_filters": ["_client.py", "_transports"], "search_scope": "medium"}}
}}
```

## Example 3: Timeout Behavior
Query: "What happens if a request exceeds the configured timeout?"
```json
{{
  "analysis": {{"query_type": "behavior", "key_terms": ["timeout", "exceed", "request"], "confidence_score": 0.88}},
  "strategy": {{"primary_search_terms": ["timeout"], "module_filters": ["_client.py", "_exceptions.py"], "search_scope": "medium"}}
}}
```"""


SAMPLE_ROUTER_DECISIONS = [
    {
        "user_query": "How does httpx validate SSL certificates?",
        "analysis": {
            "query_type": "behavior",
            "key_terms": ["ssl", "certificate", "validation", "verify"],
            "modules_of_interest": ["_ssl.py", "_client.py", "_config.py"],
            "concepts": ["SSL validation", "certificate verification"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.95
        },
        "strategy": {
            "primary_search_terms": ["ssl", "certificate", "validation"],
            "secondary_search_terms": ["verify", "cert", "tls"],
            "module_filters": ["_ssl.py", "_client.py"],
            "code_patterns": ["def.*ssl", "def.*certificate", "class.*SSL"],
            "search_scope": "medium",
            "expected_chunk_types": ["function", "method", "class"],
            "priority_modules": ["_ssl.py"]
        },
        "estimated_complexity": "medium",
        "reasoning": "This is a behavior query about SSL certificate validation, a core security feature. Focus search on SSL-related modules with both specific and general terms.",
        "potential_challenges": ["SSL validation might be split across multiple modules", "Could involve both client and transport layer code"]
    },
    {
        "user_query": "Where in the code is proxy support implemented?",
        "analysis": {
            "query_type": "location",
            "key_terms": ["proxy", "support", "implementation"],
            "modules_of_interest": ["_client.py", "_transports/", "_config.py"],
            "concepts": ["proxy support", "HTTP proxy"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.90
        },
        "strategy": {
            "primary_search_terms": ["proxy", "Proxy"],
            "secondary_search_terms": ["http_proxy", "https_proxy"],
            "module_filters": ["_client.py", "_transports/"],
            "code_patterns": ["class.*Proxy", "def.*proxy"],
            "search_scope": "medium",
            "expected_chunk_types": ["class", "function", "method"],
            "priority_modules": ["_client.py"]
        },
        "estimated_complexity": "medium",
        "reasoning": "Location query asking for proxy implementation. Focus on client and transport modules where proxy logic would be implemented.",
        "potential_challenges": ["Proxy support might be in transport adapters", "Could involve both sync and async implementations"]
    },
    {
        "user_query": "What happens if a request exceeds the configured timeout?",
        "analysis": {
            "query_type": "behavior",
            "key_terms": ["timeout", "exceed", "request", "error"],
            "modules_of_interest": ["_client.py", "_exceptions.py", "_config.py"],
            "concepts": ["timeout handling", "request timeout", "exception"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.88
        },
        "strategy": {
            "primary_search_terms": ["timeout", "Timeout"],
            "secondary_search_terms": ["exceed", "exception", "error"],
            "module_filters": ["_client.py", "_exceptions.py"],
            "code_patterns": ["class.*Timeout", "def.*timeout", "raise.*Timeout"],
            "search_scope": "medium",
            "expected_chunk_types": ["function", "class", "method"],
            "priority_modules": ["_client.py", "_exceptions.py"]
        },
        "estimated_complexity": "medium",
        "reasoning": "Behavior query about timeout behavior. Need to find timeout handling logic and exception raising code.",
        "potential_challenges": ["Timeout logic might be in multiple layers", "Could involve both client timeout and read/write timeouts"]
    }
]


def get_router_prompts() -> Dict[str, Any]:
    """Get all router prompts and examples.

    Returns:
        Dictionary containing system prompt, user template, and examples
    """
    return {
        "system_prompt": ROUTER_SYSTEM_PROMPT,
        "user_template": ROUTER_USER_PROMPT_TEMPLATE,
        "examples": SAMPLE_ROUTER_DECISIONS
    }
