"""Retriever node prompts for search execution and result processing.

This module contains prompts for the Retriever node that executes hybrid search,
applies reranking, and returns relevant code chunks with structured output.
"""

from typing import Dict, Any

__all__ = ["RETRIEVER_SYSTEM_PROMPT", "RETRIEVER_USER_PROMPT_TEMPLATE", "SAMPLE_RETRIEVAL_RESULTS"]


RETRIEVER_SYSTEM_PROMPT = """# 1. ROLE
You are a Code Search Specialist executing hybrid retrieval (semantic + keyword) for the httpx codebase.

# 2. TASK
Execute the router's retrieval strategy, perform hybrid search with reranking, and return the top-K most relevant code chunks.

# 3. CAPABILITIES
- **Hybrid search**: Dense embeddings (semantic) + sparse keywords (BM25)
- **Reranking**: Query-aware relevance scoring
- **Module filtering**: Focus on specific httpx modules
- **Fallback strategies**: Secondary terms if primary fails

# 4. OUTPUT FORMAT
Return JSON with retrieved chunks and metadata:

```json
{
  "chunks": [
    {
      "content": "code snippet",
      "file_path": "httpx/_ssl.py",
      "start_line": 123,
      "end_line": 145,
      "chunk_type": "function|class|method",
      "parent_context": "ClassName",
      "final_score": 0.88,
      "search_terms_matched": ["ssl", "validate"]
    }
  ],
  "search_metadata": {
    "chunks_retrieved": 10,
    "search_strategy": "hybrid",
    "reranking_applied": true,
    "confidence": 0.92
  },
  "reasoning": "Why these chunks are relevant",
  "next_action": "continue|re_search|synthesize"
}
```

# 5. EXECUTION STRATEGY

## Search Phases
1. **Primary Search**
   - Use primary_search_terms from router
   - Apply module_filters if specified
   - Combine dense + sparse scores (hybrid)

2. **Reranking** (if enabled)
   - Assess query-specific relevance
   - Prefer complete functions/classes
   - Ensure proper file:line citations

3. **Fallback** (if < 50% results found)
   - Try secondary_search_terms
   - Broaden module scope
   - Relax filtering criteria

## Scoring Guidelines
- **High (0.8-1.0)**: Direct answer, complete implementation
- **Medium (0.6-0.8)**: Relevant context, partial match
- **Low (< 0.6)**: Tangential relevance, filter out

# 6. QUALITY CRITERIA
✅ **Valid citations** - All chunks have file paths and line numbers
✅ **Relevant content** - Code directly relates to query
✅ **No duplicates** - Unique chunks from different locations
✅ **Diverse types** - Mix of functions, classes, methods
✅ **Complete code** - Prefer full implementations over snippets
✅ **Good documentation** - Favor chunks with docstrings


RETRIEVER_USER_PROMPT_TEMPLATE = """Execute hybrid search for the httpx codebase and return relevant code chunks.

## USER QUERY
{user_query}

## ROUTER STRATEGY
Primary terms: {primary_search_terms}
Secondary terms: {secondary_search_terms}
Module filters: {module_filters}
Search scope: {search_scope}

## SEARCH PARAMETERS
- Top-K: {top_k}
- Reranking: {reranking_enabled}

## YOUR TASK
1. **Execute primary search** with provided terms and module filters
2. **Apply hybrid scoring** (dense semantic + sparse keyword)
3. **Rerank results** if enabled (query-specific relevance)
4. **Select top-{top_k} chunks** with highest final scores
5. **Apply fallback** if < 50% results found (use secondary terms)

## OUTPUT FORMAT
Return JSON matching the structure shown in system prompt:
- `chunks`: Array of code chunks with content, file_path, line numbers, scores
- `search_metadata`: Retrieved count, strategy used, reranking status, confidence
- `reasoning`: Why these chunks are relevant to the query
- `next_action`: "continue" if good results, "re_search" if poor quality

## QUALITY CHECKS
- ✅ All chunks have valid file paths and line numbers
- ✅ Code is directly relevant to the query
- ✅ No duplicate chunks from same locations
- ✅ Prefer complete functions/classes over snippets
- ✅ Include chunks with docstrings when available

## EXAMPLES

**Query**: "How does httpx validate SSL certificates?"
**Search**: Primary ["ssl", "certificate", "validate"], modules ["_ssl.py"]
**Result**: SSL validation methods from _ssl.py, high confidence

**Query**: "Where is the Client class?"
**Search**: Primary ["Client", "class"], modules ["_client.py"], narrow scope
**Result**: Client class definition, very high confidence

Now execute the search and return the results."""


SAMPLE_RETRIEVAL_RESULTS = [
    {
        "chunks": [
            {
                "content": "def _verify_certificate_chain(self, cert, hostname):\n    \"\"\"Verify SSL certificate chain against hostname.\"\"\"\n    try:\n        self._ssl_context.check_hostname = True\n        self._ssl_context.verify_mode = ssl.CERT_REQUIRED\n        cert.verify(hostname, self._ssl_context)\n    except ssl.SSLError as e:\n        raise SSLCertVerificationError(f\"Cert verification failed: {e}\")",
                "file_path": "httpx/_ssl.py",
                "start_line": 234,
                "end_line": 245,
                "chunk_type": "method",
                "parent_context": "SSLContext",
                "final_score": 0.93,
                "search_terms_matched": ["ssl", "certificate", "validation", "verify"]
            },
            {
                "content": "def create_ssl_context(verify=True, cert=None, trust_env=True):\n    \"\"\"Create SSL context for HTTPS connections.\"\"\"\n    context = ssl.create_default_context()\n    if verify:\n        context.verify_mode = ssl.CERT_REQUIRED\n        context.check_hostname = True\n    else:\n        context.verify_mode = ssl.CERT_NONE\n    return context",
                "file_path": "httpx/_ssl.py",
                "start_line": 45,
                "end_line": 58,
                "chunk_type": "function",
                "parent_context": "",
                "final_score": 0.89,
                "search_terms_matched": ["ssl", "certificate", "verify"]
            }
        ],
        "search_metadata": {
            "chunks_retrieved": 10,
            "search_strategy": "hybrid",
            "reranking_applied": True,
            "confidence": 0.92
        },
        "reasoning": "Found direct SSL certificate validation implementation in _ssl.py. Primary chunk shows verification logic with hostname checking and CERT_REQUIRED mode. Secondary chunk shows SSL context creation. High confidence due to strong semantic and keyword matches.",
        "next_action": "continue"
    },
    {
        "chunks": [
            {
                "content": "class Client:\n    \"\"\"Synchronous HTTP client.\"\"\"\n    \n    def __init__(self, **kwargs):\n        self.headers = kwargs.get('headers', {})\n        self.timeout = kwargs.get('timeout', 5.0)\n        self.verify = kwargs.get('verify', True)\n        self._transport = HTTPTransport(**kwargs)",
                "file_path": "httpx/_client.py",
                "start_line": 89,
                "end_line": 100,
                "chunk_type": "class",
                "parent_context": "",
                "final_score": 0.96,
                "search_terms_matched": ["Client", "class"]
            }
        ],
        "search_metadata": {
            "chunks_retrieved": 1,
            "search_strategy": "hybrid",
            "reranking_applied": True,
            "confidence": 0.98
        },
        "reasoning": "Direct match for Client class definition in _client.py. Very high confidence due to exact match on class name.",
        "next_action": "continue"
    }
]


def get_retriever_prompts() -> Dict[str, Any]:
    """Get all retriever prompts and examples.

    Returns:
        Dictionary containing system prompt, user template, and examples
    """
    return {
        "system_prompt": RETRIEVER_SYSTEM_PROMPT,
        "user_template": RETRIEVER_USER_PROMPT_TEMPLATE,
        "examples": SAMPLE_RETRIEVAL_RESULTS
    }
