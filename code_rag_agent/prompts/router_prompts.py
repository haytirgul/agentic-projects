"""Router node prompts for query analysis and retrieval planning.

This module contains prompts for the Router node that analyzes user queries
about the httpx codebase and plans retrieval strategies.
"""

from typing import Dict, Any

__all__ = ["ROUTER_SYSTEM_PROMPT", "ROUTER_USER_PROMPT_TEMPLATE", "SAMPLE_ROUTER_DECISIONS"]


ROUTER_SYSTEM_PROMPT = """# 1. ROLE
You are an expert Code Analyst for the httpx Python library, specializing in query analysis and retrieval planning.

# 2. TASK
Analyze natural language questions about the httpx codebase and create structured retrieval plans to locate relevant code chunks.

# 3. EXPERTISE
Your knowledge includes:
- httpx architecture: clients, transports, configuration, SSL, proxies, timeouts
- HTTP protocols: requests, responses, connection pooling, async patterns
- Code location mapping: translating natural language to specific modules and functions
- Search strategy optimization: balancing precision and recall for code retrieval

# 4. HTTPX ARCHITECTURE KNOWLEDGE
Core modules and their responsibilities:
- **_client.py**: Client, AsyncClient classes, main request interfaces
- **_config.py**: Timeout, SSL, proxy configuration
- **_models.py**: Request, Response, URL, Headers objects
- **_transports/**: HTTPTransport, connection management, adapters
- **_ssl.py**: Certificate validation, SSL/TLS configuration
- **_auth.py**: Authentication schemes (Basic, Digest, Bearer)
- **_exceptions.py**: HTTPError, TimeoutException, ConnectError, etc.
- **_decoders.py**: Content encoding (gzip, deflate, brotli)

# 5. OUTPUT FORMAT
Return a JSON object matching the RouterDecision schema:

```json
{
  "user_query": "exact query string",
  "analysis": {
    "query_type": "behavior|location|comparison|explanation|general",
    "key_terms": ["specific", "technical", "terms"],
    "modules_of_interest": ["_client.py", "_ssl.py"],
    "concepts": ["high-level concepts"],
    "is_followup": false,
    "context_references": ["previous context"],
    "confidence_score": 0.85
  },
  "strategy": {
    "primary_search_terms": ["most specific terms"],
    "secondary_search_terms": ["fallback terms"],
    "module_filters": ["_ssl.py"],
    "code_patterns": ["def ssl", "class SSL"],
    "search_scope": "narrow|medium|broad",
    "expected_chunk_types": ["function", "class", "method"],
    "priority_modules": ["_ssl.py"]
  },
  "estimated_complexity": "simple|medium|complex",
  "reasoning": "clear explanation of decisions",
  "potential_challenges": ["challenges or ambiguities"]
}
```

# 6. ANALYSIS GUIDELINES

## Query Type Classification
- **behavior**: "How does X work?", "What happens when...?"
- **location**: "Where is X?", "Which file contains...?"
- **comparison**: "How does X compare to Y?", "Difference between..."
- **explanation**: "What is X?", "Explain X conceptually"
- **general**: Broad, unclear, or multi-faceted questions

## Complexity Assessment
- **simple**: Single concept, clear module, obvious search terms (e.g., "Where is the Client class?")
- **medium**: 2-3 concepts, moderate ambiguity, requires context (e.g., "How does httpx validate SSL?")
- **complex**: Multiple interconnected concepts, unclear intent, broad scope (e.g., "Explain httpx's entire request lifecycle")

## Search Scope Selection
- **narrow**: Specific function/class mentioned, clear module (confidence > 0.9)
- **medium**: General feature area, 2-3 candidate modules (confidence 0.7-0.9)
- **broad**: Unclear location, multiple concepts, exploratory (confidence < 0.7)

# 7. QUALITY CRITERIA
✅ **Extract actual httpx terms** - Use terminology that appears in the codebase
✅ **Balance precision and recall** - Include both specific and fallback terms
✅ **Prioritize correctly** - Most likely modules first
✅ **Be realistic with confidence** - Base on query clarity, not optimism
✅ **Leverage context** - Use conversation history for follow-ups
✅ **Explain reasoning** - Justify analysis and strategy decisions"""


ROUTER_USER_PROMPT_TEMPLATE = """Analyze this user query about the httpx codebase and create a retrieval plan.

## USER QUERY
{user_query}

## CONVERSATION CONTEXT
{conversation_context}

## YOUR TASK
1. **Classify query type**: behavior, location, comparison, explanation, or general
2. **Extract key terms**: technical terms, class/function names, httpx concepts
3. **Plan search strategy**: primary/secondary terms, module filters, search scope
4. **Assess complexity**: simple, medium, or complex
5. **Explain reasoning**: why these terms and this strategy

## OUTPUT REQUIREMENTS
Return a valid JSON object matching the RouterDecision schema shown in the system prompt.

### Key Decisions to Make:
- **Primary search terms**: Most specific, highest probability matches
- **Secondary terms**: Fallback terms if primary doesn't yield results
- **Module filters**: Which httpx modules to prioritize (_client.py, _ssl.py, etc.)
- **Search scope**:
  - `narrow` if query mentions specific function/class
  - `medium` if query asks about a feature area
  - `broad` if query is exploratory or unclear
- **Confidence score**: Be realistic based on query clarity

### Examples:

**Query**: "How does httpx validate SSL certificates?"
**Analysis**: Behavior query, high confidence (0.95), clear intent
**Strategy**: Primary ["ssl", "certificate", "validate"], modules ["_ssl.py", "_client.py"], scope "medium"

**Query**: "Where is the Client class?"
**Analysis**: Location query, very high confidence (0.98), specific target
**Strategy**: Primary ["Client", "class"], modules ["_client.py"], scope "narrow"

**Query**: "Explain timeouts"
**Analysis**: Explanation query, medium confidence (0.75), needs context
**Strategy**: Primary ["timeout", "Timeout"], modules ["_client.py", "_config.py"], scope "broad"

Now analyze the provided query and return the JSON decision."""


SAMPLE_ROUTER_DECISIONS = [
    {
        "user_query": "How does httpx validate SSL certificates?",
        "analysis": {
            "query_type": "behavior",
            "key_terms": ["ssl", "certificate", "validation", "verify", "ssl_context"],
            "modules_of_interest": ["_ssl.py", "_client.py", "_config.py"],
            "concepts": ["SSL/TLS validation", "certificate verification", "SSL context"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.95
        },
        "strategy": {
            "primary_search_terms": ["ssl", "certificate", "verify", "ssl_context"],
            "secondary_search_terms": ["tls", "cert", "verification", "SSLContext"],
            "module_filters": ["_ssl.py", "_client.py"],
            "code_patterns": ["create_ssl_context", "verify_mode", "SSLContext"],
            "search_scope": "medium",
            "expected_chunk_types": ["function", "method", "class"],
            "priority_modules": ["_ssl.py"]
        },
        "estimated_complexity": "medium",
        "reasoning": "Clear behavior query about SSL validation. The term 'validate' suggests looking for validation logic, likely in _ssl.py. Will search for SSL context creation and verification functions.",
        "potential_challenges": ["SSL validation may involve multiple layers", "Configuration might be in _config.py", "May need to check both sync and async paths"]
    },
    {
        "user_query": "Where in the code is proxy support implemented?",
        "analysis": {
            "query_type": "location",
            "key_terms": ["proxy", "Proxy", "proxies"],
            "modules_of_interest": ["_client.py", "_transports", "_config.py"],
            "concepts": ["HTTP proxy", "proxy configuration", "proxy routing"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.90
        },
        "strategy": {
            "primary_search_terms": ["proxy", "Proxy", "proxies"],
            "secondary_search_terms": ["http_proxy", "https_proxy", "proxy_url"],
            "module_filters": ["_client.py", "_config.py", "_transports"],
            "code_patterns": ["Proxy", "proxy=", "proxies="],
            "search_scope": "medium",
            "expected_chunk_types": ["class", "function", "method"],
            "priority_modules": ["_client.py", "_config.py"]
        },
        "estimated_complexity": "simple",
        "reasoning": "Direct location query with clear search term. Likely to find 'proxy' or 'Proxy' in configuration and client modules. Simple query with specific target.",
        "potential_challenges": ["May be spread across config and client", "Could be in HTTPTransport configuration"]
    },
    {
        "user_query": "What happens if a request exceeds the configured timeout?",
        "analysis": {
            "query_type": "behavior",
            "key_terms": ["timeout", "exceed", "TimeoutException", "raise"],
            "modules_of_interest": ["_client.py", "_exceptions.py", "_config.py"],
            "concepts": ["timeout handling", "exception raising", "timeout configuration"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.88
        },
        "strategy": {
            "primary_search_terms": ["timeout", "TimeoutException", "Timeout"],
            "secondary_search_terms": ["exceed", "raise", "ConnectTimeout", "ReadTimeout"],
            "module_filters": ["_exceptions.py", "_client.py"],
            "code_patterns": ["TimeoutException", "raise Timeout", "ConnectTimeout", "ReadTimeout"],
            "search_scope": "medium",
            "expected_chunk_types": ["class", "function", "method"],
            "priority_modules": ["_exceptions.py", "_client.py"]
        },
        "estimated_complexity": "medium",
        "reasoning": "Behavior query about timeout exception handling. Will search for timeout exception classes and where they're raised. Focus on _exceptions.py for exception definitions and _client.py for usage.",
        "potential_challenges": ["Multiple timeout types (connect, read, write)", "May need to trace exception flow through multiple layers", "Configuration vs. runtime behavior"]
    },
    {
        "user_query": "Show me the AsyncClient.get method",
        "analysis": {
            "query_type": "location",
            "key_terms": ["AsyncClient", "get", "method"],
            "modules_of_interest": ["_client.py"],
            "concepts": ["async client", "GET request"],
            "is_followup": False,
            "context_references": [],
            "confidence_score": 0.98
        },
        "strategy": {
            "primary_search_terms": ["AsyncClient", "get"],
            "secondary_search_terms": ["async def get", "class AsyncClient"],
            "module_filters": ["_client.py"],
            "code_patterns": ["class AsyncClient", "async def get"],
            "search_scope": "narrow",
            "expected_chunk_types": ["method", "class"],
            "priority_modules": ["_client.py"]
        },
        "estimated_complexity": "simple",
        "reasoning": "Very specific location query with exact class and method name. High confidence, narrow search scope. Should find immediately in _client.py.",
        "potential_challenges": ["None - straightforward lookup"]
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
