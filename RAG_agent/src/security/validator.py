"""High-level security validation API with LangChain text splitting.

This module provides the main entry point for validating user inputs
using LangChain's RecursiveCharacterTextSplitter for optimal chunking.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .exceptions import SecurityValidationError
from .prompt_guard import PromptGuardClassifier

logger = logging.getLogger(__name__)


class SecurityValidator:
    """High-level security validator with LangChain text splitting.

    This class orchestrates the validation workflow:
    1. Split text using LangChain RecursiveCharacterTextSplitter
    2. Classify all chunks in batch using Prompt Guard (OPTIMIZED)
    3. Aggregate results (fail if ANY chunk is malicious)

    Attributes:
        classifier: PromptGuardClassifier instance
        text_splitter: LangChain text splitter instance
        fail_on_first: Stop validation on first malicious chunk (faster)
    """

    def __init__(
        self,
        malicious_threshold: float = 0.5,
        fail_on_first: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """Initialize the security validator.

        Args:
            malicious_threshold: Confidence threshold for malicious classification
            fail_on_first: Stop on first malicious chunk (faster but less detailed)
            device: Device for ML model ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for classifier (default: 32)
        """
        self.classifier = PromptGuardClassifier(
            malicious_threshold=malicious_threshold, device=device, batch_size=batch_size
        )

        # Use RecursiveCharacterTextSplitter with conservative splitting
        # Prioritize keeping complex queries intact when possible
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Maximum 512 characters per chunk (model limitation)
            chunk_overlap=50,  # Overlap to maintain context across splits
            separators=[
                "\n\n",        # Paragraph breaks (highest priority)
                "\n",          # Line breaks (preserve structure)
                # Be very conservative with sentence splitting to avoid breaking complex queries
                "? ",          # Questions (clear sentence endings)
                "! ",          # Exclamations (clear sentence endings)
                # Put period splitting much lower to avoid breaking quoted text
                "; ",          # Semi-colons (clause boundaries)
                ": ",          # Colons (but avoid '::' patterns)
                ". ",          # Periods (lowest sentence priority - often breaks complex text)
                # Very low priority separators
                ", ",          # Commas (clause level)
                " ",           # Words (last resort for semantic splitting)
                "",            # Characters (absolute fallback)
            ],
            length_function=len,
            keep_separator=False,
        )

        self.fail_on_first = fail_on_first

        logger.info(
            f"SecurityValidator initialized: semantic_splitting=True, "
            f"threshold={malicious_threshold}, batch_size={batch_size}"
        )


    def _smart_split_text(self, text: str) -> List[str]:
        """Smart text splitting that tries to preserve semantic meaning.

        Uses a two-phase approach:
        1. Try conservative splitting (paragraphs/line breaks only)
        2. If chunks are still too long, fall back to sentence-aware splitting

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        # Phase 1: Try splitting only on major structural boundaries
        conservative_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=25,
            separators=["\n\n", "\n"],  # Only paragraph and line breaks
            length_function=len,
        )

        chunks = conservative_splitter.split_text(text)

        # Check if all chunks are within size limits
        if all(len(chunk) <= 512 for chunk in chunks):
            logger.debug(f"Conservative splitting successful: {len(chunks)} chunks")
            return chunks

        # Phase 2: Fall back to full semantic splitting
        logger.debug("Conservative splitting failed, using full semantic splitting")
        return self.text_splitter.split_text(text)

    def _validate_uncached(
        self, text: str
    ) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """Internal validation logic without caching.

        Args:
            text: User input to validate

        Returns:
            Tuple of (is_safe, reason, chunk_results)
        """
        if not text or not text.strip():
            return True, None, []

        # Use smart splitting to preserve semantic meaning
        chunks = self._smart_split_text(text)

        if not chunks:
            # No chunks produced, validate whole text
            chunks = [text]

        logger.debug(f"Smart text splitting: {len(chunks)} chunks created")
        if logger.isEnabledFor(logging.DEBUG):
            for i, chunk in enumerate(chunks):
                logger.debug(f"Chunk {i}: {len(chunk)} chars - '{chunk[:100]}{'...' if len(chunk) > 100 else ''}'")

        # BATCH CLASSIFICATION - classify all chunks at once for optimal performance
        classification_results = self.classifier.batch_classify(chunks)

        # Build chunk results with metadata
        chunk_results: List[Dict[str, Any]] = []
        malicious_chunks: List[Tuple[int, str, Dict[str, Any]]] = []

        for i, (chunk, result) in enumerate(zip(chunks, classification_results)):
            chunk_results.append(
                {"chunk_index": i, "chunk": chunk[:100], "result": result}
            )

            if not result["is_safe"]:
                malicious_chunks.append((i, chunk, result))
                logger.info(
                    f"Malicious chunk detected: chunk={i}, "
                    f"label={result['label']}, confidence={result['confidence']:.2%}"
                )

                # Fail fast if enabled
                if self.fail_on_first:
                    break

        # Aggregate results
        is_safe = len(malicious_chunks) == 0

        if is_safe:
            logger.info(f"Input validated as safe ({len(chunks)} chunks)")
            return True, None, chunk_results

        # Build detailed reason
        reason = self._build_failure_reason(malicious_chunks)
        return False, reason, chunk_results

    def validate(
        self, 
        text: str,
        raise_on_malicious: bool = True
    ) -> Tuple[bool, Optional[str], List[Dict[str, Any]]]:
        """Validate user input for security threats.

        This method:
        1. Splits input using semantic boundary splitting (paragraphs, sentences, clauses)
        2. Classifies all semantic units in batch (OPTIMIZED)
        3. Aggregates results (fails if ANY unit is malicious)
        4. Optionally raises exception on malicious content

        Args:
            text: User input to validate
            raise_on_malicious: Raise SecurityValidationError if malicious detected

        Returns:
            Tuple of (is_safe, reason, chunk_results)
                - is_safe: False if any semantic unit is malicious
                - reason: Human-readable explanation (None if safe)
                - chunk_results: List of per-semantic-unit classification results

        Raises:
            SecurityValidationError: If malicious content detected and raise_on_malicious=True
        """
        is_safe, reason, chunk_results = self._validate_uncached(text)

        if not is_safe and raise_on_malicious:
            # Get highest confidence malicious result
            malicious_results = [
                chunk["result"] for chunk in chunk_results if not chunk["result"]["is_safe"]
            ]
            max_confidence = (
                max(r["confidence"] for r in malicious_results) if malicious_results else 0.0
            )

            raise SecurityValidationError(
                message=reason or "Security threat detected", confidence=max_confidence, detected_type="prompt_injection"
            )

        return is_safe, reason, chunk_results

    def _build_failure_reason(self, malicious_chunks: List[Tuple[int, str, Dict[str, Any]]]) -> str:
        """Build human-readable failure message.

        Args:
            malicious_chunks: List of (index, chunk, result) tuples

        Returns:
            Formatted error message
        """
        num_malicious = len(malicious_chunks)

        if num_malicious == 1:
            idx, chunk, result = malicious_chunks[0]
            return (
                f"Security threat detected in input (confidence: {result['confidence']:.2%}). "
                f"Potential prompt injection or jailbreak attempt."
            )
        else:
            max_confidence = max(r["confidence"] for _, _, r in malicious_chunks)
            return (
                f"Security threats detected in {num_malicious} segments "
                f"(confidence: {max_confidence:.2%}). "
                f"Potential prompt injection or jailbreak attempts."
            )

    def batch_validate(
        self, texts: List[str], raise_on_malicious: bool = True
    ) -> List[Tuple[bool, Optional[str], List[Dict[str, Any]]]]:
        """Validate multiple inputs.

        Note: While this validates multiple texts, each text is still chunked
        and its chunks are validated in batch. For optimal performance,
        prefer validating longer texts that will be split into multiple chunks.

        Args:
            texts: List of user inputs to validate
            raise_on_malicious: Raise exception on first malicious input

        Returns:
            List of validation results (same format as validate())

        Raises:
            SecurityValidationError: If malicious content detected and raise_on_malicious=True
        """
        results = []

        for text in texts:
            try:
                result = self.validate(text, raise_on_malicious=raise_on_malicious)
                results.append(result)
            except SecurityValidationError:
                if raise_on_malicious:
                    raise
                # Should not reach here, but handle gracefully
                results.append((False, "Security validation failed", []))

        return results


# Singleton instance for convenience
_default_validator: Optional[SecurityValidator] = None


def get_default_validator() -> SecurityValidator:
    """Get or create the default SecurityValidator instance.

    Returns:
        Singleton SecurityValidator
    """
    global _default_validator

    if _default_validator is None:
        _default_validator = SecurityValidator()

    return _default_validator


def validate_user_input(
    text: str, 
    raise_on_malicious: bool = True
) -> Tuple[bool, Optional[str]]:
    """Convenience function for quick validation (uses default validator).

    Args:
        text: User input to validate
        raise_on_malicious: Raise exception if malicious detected

    Returns:
        Tuple of (is_safe, reason)

    Raises:
        SecurityValidationError: If malicious content detected and raise_on_malicious=True

    Example:
        >>> from src.security import validate_user_input
        >>> is_safe, reason = validate_user_input("I want to order 3 apples")
        >>> if not is_safe:
        ...     print(f"Blocked: {reason}")
    """
    validator = get_default_validator()
    if not validator.classifier.is_model_available():
        return True, "Security validation unavailable (degraded mode)"
    is_safe, reason, _ = validator.validate(text, raise_on_malicious=raise_on_malicious)
    return is_safe, reason
