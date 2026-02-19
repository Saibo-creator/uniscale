"""
HuggingFace Tokenizers backend.

This backend uses the HuggingFace tokenizers library for training.
Supports: BPE (Byte-level BPE)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
)
from transformers import PreTrainedTokenizerFast

from uniscale.tokenizers.backends.base import TokenizerBackend


class HuggingFaceBackend(TokenizerBackend):
    """
    Backend for HuggingFace tokenizers library.

    Supports:
    - bpe: Byte-level BPE tokenization
    """

    def get_supported_algorithms(self) -> List[str]:
        return ["bpe"]

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tokenizer using HuggingFace tokenizers library.

        Args:
            algorithm: Must be "bpe"
            data_file: Path to training data (JSONL)
            vocab_size: Target vocabulary size
            output_dir: Directory to save tokenizer
            min_frequency: Minimum frequency for tokens
            special_tokens: List of special tokens
            **kwargs: Additional parameters

        Returns:
            Dictionary with training metadata
        """
        if algorithm not in self.get_supported_algorithms():
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by HuggingFaceBackend. "
                f"Supported: {self.get_supported_algorithms()}"
            )

        if special_tokens is None:
            special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]

        print(f"Training BPE tokenizer with vocab_size={vocab_size}")

        # Initialize tokenizer with ByteLevel BPE
        tokenizer = Tokenizer(models.BPE())

        # Use consistent pretokenization: ByteLevel
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Normalization (minimal for byte-level)
        tokenizer.normalizer = normalizers.Sequence([])

        # Post-processing
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        # Train on iterator
        print("Reading training data...")
        tokenizer.train_from_iterator(
            self.get_text_iterator(data_file),
            trainer=trainer,
            length=None,
        )

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path / "tokenizer.json"))

        print(f"✓ Saved BPE tokenizer to {output_path}")

        # Save config
        config = {
            "backend": "huggingface",
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "special_tokens": special_tokens,
        }
        with open(output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def export_to_hf(
        self,
        artifacts_dir: str,
        output_dir: str,
        **kwargs
    ) -> PreTrainedTokenizerFast:
        """
        Export HuggingFace tokenizer (already in HF format).

        Args:
            artifacts_dir: Directory containing tokenizer.json
            output_dir: Directory to save HF tokenizer
            **kwargs: Additional parameters

        Returns:
            Loaded HuggingFace tokenizer
        """
        artifacts_path = Path(artifacts_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        tokenizer_file = artifacts_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))

        # Save in HF format
        tokenizer.save_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")

        return tokenizer
