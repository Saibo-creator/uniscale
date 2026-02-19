"""
Abstract base class for tokenizer training backends.

This module defines the interface that all tokenizer backends must implement,
allowing for seamless integration of different tokenizer training libraries.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
from transformers import PreTrainedTokenizerFast


class TokenizerBackend(ABC):
    """
    Abstract base class for tokenizer training backends.

    Each backend wraps a specific tokenizer training library (e.g., HuggingFace tokenizers,
    SentencePiece, parity-aware-bpe, TkTkT) and provides a unified interface for:
    1. Training tokenizers with specific algorithms
    2. Exporting trained tokenizers to HuggingFace format
    """

    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        """
        Return a list of tokenization algorithms supported by this backend.

        Returns:
            List of algorithm names (e.g., ["bpe", "unigram"])
        """
        pass

    @abstractmethod
    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tokenizer using this backend.

        Args:
            algorithm: The tokenization algorithm to use (must be in get_supported_algorithms())
            data_file: Path to training data (JSONL format with 'text' field)
            vocab_size: Target vocabulary size
            output_dir: Directory to save training artifacts
            **kwargs: Additional backend-specific parameters

        Returns:
            Dictionary containing training metadata and paths to artifacts
        """
        pass

    @abstractmethod
    def export_to_hf(
        self,
        artifacts_dir: str,
        output_dir: str,
        **kwargs
    ) -> PreTrainedTokenizerFast:
        """
        Export trained tokenizer to HuggingFace format.

        Args:
            artifacts_dir: Directory containing training artifacts
            output_dir: Directory to save HF tokenizer
            **kwargs: Additional export parameters

        Returns:
            Loaded HuggingFace tokenizer
        """
        pass

    def get_text_iterator(
        self,
        data_file: str,
        batch_size: int = 1000
    ) -> Iterator[List[str]]:
        """
        Default text iterator for JSONL files.

        Yields batches of text from JSONL file with 'text' field.
        Backends can override this if they need different data formats.

        Args:
            data_file: Path to JSONL file
            batch_size: Number of texts per batch

        Yields:
            Batches of text strings
        """
        import json

        batch = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("text", "")
                if text:
                    batch.append(text)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

        if batch:
            yield batch
