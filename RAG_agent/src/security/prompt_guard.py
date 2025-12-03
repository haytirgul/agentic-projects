"""ML-based prompt injection detection using ProtectAI DeBERTa v3.

This module wraps the ProtectAI DeBERTa v3 base model for fast, efficient
detection of prompt injection and jailbreak attacks with batch optimization.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

from settings import (
    SECURITY_FAIL_CLOSED,
    SECURITY_MAX_BATCH_SIZE,
    SECURITY_MODEL_CACHE_DIR,
    SECURITY_USE_FP16,
)

from .exceptions import SecurityWarning

logger = logging.getLogger(__name__)


class PromptGuardClassifier:
    """Wrapper for prompt injection detection.

    This classifier uses ProtectAI DeBERTa v3 base model for prompt injection detection.

    All classification is done in batches for optimal performance.

    Attributes:
        classifier: Transformers pipeline for text classification
        tokenizer: Tokenizer instance for text preprocessing
        model: Model instance for classification
        device: Device (CPU/CUDA) for model inference
        malicious_threshold: Confidence threshold for classifying as malicious
        is_available: Whether model loaded successfully
        batch_size: Default batch size for batch processing
        fail_closed: Whether to fail closed (block on errors) or fail open
        cache_dir: Directory for model cache
        use_fp16: Whether to use FP16 precision on CUDA
    """

    # Model configuration
    MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
    MALICIOUS_LABEL = "INJECTION"
    BENIGN_LABEL = "SAFE"
    MAX_BATCH_SIZE = SECURITY_MAX_BATCH_SIZE

    def __init__(
        self,
        malicious_threshold: float = 0.75,
        device: Optional[str] = None,
        batch_size: int = 32,
        fail_closed: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ):
        """Initialize the Prompt Guard classifier.

        Loads ProtectAI DeBERTa v3 base model for prompt injection detection.

        Args:
            malicious_threshold: Confidence threshold for malicious classification (0.0-1.0)
            device: Device to run model on ('cpu', 'cuda', or None for auto)
            batch_size: Default batch size for processing (default: 32)
            fail_closed: Fail closed on errors (True) or fail open (False). If None, uses settings
            cache_dir: Directory for model cache. If None, uses settings or default HF cache
            use_fp16: Use FP16 precision on CUDA. If None, uses settings

        Raises:
            ImportError: If transformers library not installed
        """
        self.malicious_threshold = malicious_threshold
        self.batch_size = batch_size
        self.classifier = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.is_available = False

        # Configuration from settings or parameters
        self.fail_closed = fail_closed if fail_closed is not None else SECURITY_FAIL_CLOSED
        self.cache_dir = cache_dir if cache_dir is not None else SECURITY_MODEL_CACHE_DIR
        self.use_fp16 = use_fp16 if use_fp16 is not None else SECURITY_USE_FP16

        logger.info(
            f"Initializing PromptGuardClassifier: "
            f"threshold={malicious_threshold}, "
            f"batch_size={batch_size}, "
            f"fail_closed={self.fail_closed}"
        )

        try:
            self.load_model(device)
        except ImportError as e:
            warnings.warn(
                "transformers library not installed. Security validation disabled. "
                "Install with: pip install transformers torch",
                SecurityWarning,
            )
            logger.warning(f"Failed to import transformers: {e}")

    def load_model(self, device: Optional[str] = None) -> None:
        """Load the model with explicit control over tokenizer, model, and device.

        Args:
            device: Device to run model on ('cpu', 'cuda', or None for auto-detect)

        Raises:
            ImportError: If transformers library not available
            Exception: If model fails to load
        """
        logger.info(f"Loading model: {self.MODEL}")

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        # Prepare common kwargs for tokenizer and model
        common_kwargs: Dict[str, Any] = {}

        # Add cache directory if specified
        if self.cache_dir:
            common_kwargs["cache_dir"] = self.cache_dir
            logger.info(f"Using cache directory: {self.cache_dir}")
        else:
            logger.debug("Using default Hugging Face cache directory")

        # Load tokenizer
        try:
            logger.debug("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL,
                **common_kwargs
            )
            logger.debug("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}", exc_info=True)
            raise

        # Load model
        try:
            logger.debug("Loading model...")
            model_kwargs = common_kwargs.copy()

            # Use FP16 precision if enabled and on CUDA
            if self.use_fp16 and self.device.type == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Using FP16 precision for faster inference")
            else:
                if self.use_fp16 and self.device.type != "cuda":
                    logger.warning(
                        "FP16 requested but not on CUDA device. Using default precision."
                    )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL,
                **model_kwargs
            )
            self.model.to(self.device)
            logger.debug(f"Model moved to device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

        # Create pipeline
        try:
            logger.debug("Creating classification pipeline...")
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device.type == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            self.is_available = True
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}", exc_info=True)
            raise

    def classify(self, text: str) -> Dict[str, Any]:
        """Classify single text as benign or malicious.

        For single inputs, internally uses batch processing for consistency.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with:
                - is_safe (bool): False if malicious detected
                - label (str): 'MALICIOUS' or 'BENIGN'
                - confidence (float): Model confidence score (0.0-1.0)
                - reason (str): Human-readable explanation
        """
        results = self.batch_classify([text])
        return results[0]

    def is_model_available(self) -> bool:
        """Check if the model is available."""
        return self.is_available and self.classifier is not None

    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple texts in batch (OPTIMIZED for performance).

        This is the primary method for classification - even single inputs
        go through this for consistency and to leverage batch optimization.

        Args:
            texts: List of input texts to classify

        Returns:
            List of classification results, one per input text

        Raises:
            TypeError: If texts is not a list or contains non-string items
            ValueError: If batch size exceeds maximum allowed
        """
        # Input validation: Check type
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")

        # Input validation: Check batch size
        if len(texts) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(texts)} exceeds maximum {self.MAX_BATCH_SIZE}. "
                f"Process in smaller batches."
            )

        # Input validation: Check text items are strings
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Text at index {idx} must be string, got {type(text).__name__}"
                )

        # Handle empty texts
        results = []
        valid_texts = []
        valid_indices = []

        for idx, text in enumerate(texts):
            if not text or not text.strip():
                results.append(
                    {
                        "is_safe": True,
                        "label": self.BENIGN_LABEL,
                        "confidence": 1.0,
                        "reason": "Empty input",
                    }
                )
            else:
                valid_texts.append(text[:512])  # Truncate to max length
                valid_indices.append(idx)
                results.append({})  # Placeholder to be filled later

        if not valid_texts:
            return results

        try:
            # BATCH CLASSIFICATION - optimized for performance
            logger.debug(f"Batch classifying {len(valid_texts)} texts")
            predictions = self.classifier(valid_texts)

            # Process results
            for valid_idx, prediction in zip(valid_indices, predictions):
                label = prediction["label"]
                confidence = prediction["score"]

                logger.debug(f"Model prediction: label='{label}', score={confidence:.4f}")

                # Determine safety based on label and threshold
                is_malicious = (
                    label == self.MALICIOUS_LABEL
                    and confidence >= self.malicious_threshold
                )

                logger.debug(f"Classification result: is_malicious={is_malicious}")

                if is_malicious:
                    reason = (
                        f"Potential prompt injection or jailbreak detected "
                        f"(confidence: {confidence:.2%})"
                    )
                else:
                    reason = "Input validated as safe"

                results[valid_idx] = {
                    "is_safe": not is_malicious,
                    "label": label,
                    "confidence": confidence,
                    "reason": reason,
                }

            return results

        except Exception as e:
            logger.error(f"Error during batch classification: {e}", exc_info=True)

            # Apply fail-closed or fail-open policy
            safe_value = not self.fail_closed

            if self.fail_closed:
                logger.warning(
                    "Fail-closed policy: Blocking request due to classification error"
                )
            else:
                logger.warning(
                    "Fail-open policy: Allowing request through despite classification error"
                )

            return [
                {
                    "is_safe": safe_value,
                    "label": "ERROR",
                    "confidence": 0.0,
                    "reason": f"Security check failed: {str(e)}",
                }
                for _ in texts
            ]


