"""Security module for prompt injection detection and input validation.

This module provides ML-based security validation using ProtectAI DeBERTa v3
to protect against prompt injection attacks, jailbreak attempts, and malicious content.

Author: Hay Hoffman
Version: 1.1
"""

from src.security.exceptions import SecurityValidationError, SecurityWarning
from src.security.validator import SecurityValidator, get_default_validator, validate_user_input
from src.security.prompt_guard import PromptGuardClassifier

__all__ = [
    "SecurityValidationError",
    "SecurityWarning",
    "SecurityValidator",
    "PromptGuardClassifier",
    "get_default_validator",
    "validate_user_input",
]
