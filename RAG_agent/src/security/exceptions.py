"""Custom exceptions for security validation."""


class SecurityValidationError(Exception):
    """Raised when input fails security validation (malicious content detected)."""

    def __init__(self, message: str, confidence: float = 0.0, detected_type: str = "unknown"):
        """Initialize security validation error.

        Args:
            message: Human-readable error message
            confidence: ML model confidence score (0.0-1.0)
            detected_type: Type of threat detected (e.g., 'injection', 'jailbreak')
        """
        super().__init__(message)
        self.confidence = confidence
        self.detected_type = detected_type


class SecurityWarning(Warning):
    """Warning for non-critical security issues or degraded mode."""

    pass
