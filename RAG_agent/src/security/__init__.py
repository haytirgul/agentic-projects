"""Security validation module for LLM prompt injection and jailbreak detection.

This module provides ML-based security validation using ProtectAI DeBERTa v3 Base
to detect malicious content, prompt injection attempts, and jailbreak attacks in user inputs.

Exports:
    validate_user_input: High-level validation function
    SecurityValidator: Main validator class with caching and batch optimization
    PromptGuardClassifier: ML model wrapper with batch processing
    SecurityValidationError: Exception for security violations
"""

from .exceptions import SecurityValidationError, SecurityWarning
from .prompt_guard import PromptGuardClassifier
from .validator import SecurityValidator, validate_user_input

__all__ = [
    "validate_user_input",
    "SecurityValidator",
    "PromptGuardClassifier",
    "SecurityValidationError",
    "SecurityWarning",
]
