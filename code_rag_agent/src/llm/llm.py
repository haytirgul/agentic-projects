"""
LLM factory with tool binding and structured output support.

This module provides the core LLM instantiation with configurable features.
It serves as the single source of truth for LLM configuration.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from settings import (
    GOOGLE_API_KEY,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_K,
    LLM_TOP_P,
    MODEL_FAST,
    MODEL_INTERMEDIATE,
    MODEL_SLOW,
)

__all__ = ["create_llm_with_config", "get_cached_llm", "initialize_llm_cache"]

logger = logging.getLogger(__name__)

# LLM Cache System
# Global cache for LLM instances to avoid recreation on every iteration
_LLM_CACHE: dict[str, BaseChatModel] = {}


def get_cached_llm(model: str) -> BaseChatModel:
    """
    Get a cached LLM instance for a specific model.

    Args:
        model: Model name

    Returns:
        Cached LLM instance

    Raises:
        ValueError: If model is not in cache
    """
    if model not in _LLM_CACHE:
        raise ValueError(f"Model '{model}' not found in cache. Initialize LLM cache first.")

    return _LLM_CACHE[model]


def initialize_llm_cache() -> None:
    """
    Pre-create and cache all model instances used in the graph.
    Call this once during graph initialization.

    Caches:
    - MODEL_FAST (gemini-2.5-flash-lite) - for routing
    - MODEL_INTERMEDIATE (gemini-2.5-flash) - for synthesis
    - MODEL_SLOW (gemini-2.5-pro) - for complex tasks
    """

    def _cache_model(model: str) -> None:
        """Cache a single model with default settings."""
        if model not in _LLM_CACHE:
            _LLM_CACHE[model] = create_llm_with_config(model=model)
            logger.info(f"Created and cached LLM: {model}")

    logger.info("Initializing LLM cache...")

    # Models to cache (deduplicate in case of overlap)
    models_to_cache = list(set([MODEL_FAST, MODEL_INTERMEDIATE, MODEL_SLOW]))

    # Cache models in parallel
    with ThreadPoolExecutor(max_workers=len(models_to_cache)) as executor:
        futures = [executor.submit(_cache_model, model) for model in models_to_cache]

        # Wait for all to complete
        for future in as_completed(futures):
            future.result()  # Raise any exceptions

    logger.info(f"LLM cache initialized with {len(_LLM_CACHE)} model instances")



def create_llm_with_config(
    model: str,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int | None = LLM_MAX_TOKENS,
    top_p: float = LLM_TOP_P,
    top_k: int = LLM_TOP_K,
    timeout: int = LLM_TIMEOUT,
    max_retries: int = LLM_MAX_RETRIES,
) -> BaseChatModel:
    """
    Create a ChatGoogleGenerativeAI instance with specified configuration.

    This centralizes LLM creation logic to avoid code duplication across nodes.

    Args:
        model: The model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
        temperature: Temperature setting for randomness (0.0 to 1.0)
        max_tokens: Maximum output tokens (None for model default)
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Raises:
        RuntimeError: If LLM initialization fails

    Example:
        >>> llm = create_llm_with_config(
        ...     model="gemini-2.0-flash",
        ...     temperature=0.1,
        ...     max_tokens=4096
        ... )
    """
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.error(f"Failed to create LLM instance: {e}")
        raise RuntimeError(f"Failed to initialize LLM: {e}") from e




