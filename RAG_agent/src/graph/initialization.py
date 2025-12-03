"""
System initialization module for LLM cache and RAG components.

This module handles the parallel initialization of:
1. LLM Cache: All Gemini model instances
2. RAG Components: PKL files (DocumentIndex, BM25, Fuzzy matcher)
3. VectorDB: ChromaDB with embeddings

The initialization is thread-safe, idempotent, and optimized for performance.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["initialize_system"]

logger = logging.getLogger(__name__)

# Track if system has been initialized
_system_initialized = False
_init_lock = threading.Lock()


def initialize_system() -> None:
    """
    Initialize all system components: LLM cache, RAG, and Security models.

    This function can be called manually during application startup to pre-load
    components, or it will be called automatically by build_agent_graph().

    This performs parallel initialization of:
    1. LLM Cache: Loads all 3 Gemini model instances
    2. RAG Components: Loads PKL files (DocumentIndex, BM25, Fuzzy matcher) + VectorDB
    3. Security: Loads ProtectAI DeBERTa v3 model (if HF_TOKEN available)

    The function is:
    - Thread-safe: Uses a lock to prevent concurrent initialization
    - Idempotent: Safe to call multiple times (only initializes once)
    - Parallel: All components load simultaneously for maximum speed

    Raises:
        RuntimeError: If LLM or RAG initialization fails
        FileNotFoundError: If RAG component files are missing

    Note:
        Security model failure is non-fatal - system runs in degraded mode without it.

    Example:
        >>> from src.graph.initialization import initialize_system
        >>> initialize_system()  # Pre-initialize before building graph
        >>> # Later...
        >>> from src.graph.builder import build_agent_graph
        >>> graph = build_agent_graph()  # Components already initialized
    """
    global _system_initialized

    with _init_lock:
        if _system_initialized:
            logger.info("System already initialized, skipping...")
            return

        logger.info("=" * 60)
        logger.info("INITIALIZING SYSTEM COMPONENTS")
        logger.info("=" * 60)

        # Import here to avoid circular dependencies
        from src.llm import initialize_llm_cache
        from src.nodes.hybrid_retrieval_vector import initialize_rag_components
        from settings import SECURITY_ENABLED

        # Track initialization errors (security is optional, so separate tracking)
        init_errors = {"llm": None, "rag": None}
        security_initialized = False

        def init_llm():
            """Initialize LLM cache."""
            try:
                logger.info("Starting LLM initialization...")
                initialize_llm_cache()
                logger.info("✓ LLM cache initialized")
            except Exception as e:
                init_errors["llm"] = e
                logger.error(f"✗ LLM initialization failed: {e}")

        def init_rag():
            """Initialize RAG components."""
            try:
                logger.info("Starting RAG initialization (PKL + VectorDB)...")
                initialize_rag_components()
                logger.info("✓ RAG components initialized")
            except Exception as e:
                init_errors["rag"] = e
                logger.error(f"✗ RAG initialization failed: {e}")

        def init_security():
            """Initialize security validator (non-fatal if fails)."""
            nonlocal security_initialized
            try:
                if not SECURITY_ENABLED:
                    logger.info("⚠ Security validation DISABLED (set SECURITY_ENABLED=true to enable)")
                    return

                logger.info("Starting Security initialization (ProtectAI DeBERTa v3)...")
                # Import here to avoid loading transformers unless needed
                from src.security.validator import get_default_validator

                # This triggers loading of the Prompt Guard model
                validator = get_default_validator()
                if validator.classifier.is_available:
                    logger.info("✓ Security validator initialized (ML-based)")
                    security_initialized = True
                else:
                    logger.info("⚠ Security validator in degraded mode (no ML model)")
            except Exception as e:
                logger.warning(f"⚠ Security initialization failed (degraded mode): {e}")

        # Run all initializations in parallel for maximum speed
        llm_thread = threading.Thread(target=init_llm, name="LLM-Init")
        rag_thread = threading.Thread(target=init_rag, name="RAG-Init")

        # Only start security thread if enabled
        if SECURITY_ENABLED:
            security_thread = threading.Thread(target=init_security, name="Security-Init")
            security_thread.start()
        else:
            # Run inline to log the disabled message
            init_security()

        llm_thread.start()
        rag_thread.start()

        # Wait for all to complete
        llm_thread.join()
        rag_thread.join()
        if SECURITY_ENABLED:
            security_thread.join()

        # Check for fatal initialization errors (LLM and RAG are required)
        if init_errors["llm"]:
            raise RuntimeError(
                f"LLM initialization failed: {init_errors['llm']}"
            ) from init_errors["llm"]

        if init_errors["rag"]:
            raise RuntimeError(
                f"RAG initialization failed: {init_errors['rag']}\n"
                f"Hint: Run 'python scripts/data_pipeline/indexing/build_rag_components.py' to build RAG indices"
            ) from init_errors["rag"]

        logger.info("=" * 60)
        logger.info("✓ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        if not security_initialized:
            logger.info("  (Security running in degraded mode - set HF_TOKEN to enable)")
        logger.info("=" * 60)

        _system_initialized = True
