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
    - unigram: Unigram language model tokenization
    """

    def get_supported_algorithms(self) -> List[str]:
        return ["bpe", "unigram"]

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
        pre_tokenizer: str = None,
        # Unigram-specific parameters
        max_piece_length: int = 16,
        shrinking_factor: float = 0.75,
        n_sub_iterations: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tokenizer using HuggingFace tokenizers library.

        Args:
            algorithm: "bpe" or "unigram"
            data_file: Path to training data (JSONL)
            vocab_size: Target vocabulary size
            output_dir: Directory to save tokenizer
            min_frequency: Minimum frequency for tokens (BPE only)
            special_tokens: List of special tokens
            pre_tokenizer: Pretokenization strategy: "byte_level" (default), "apertus", "gpt2", "gpt4", or "whitespace"
            max_piece_length: Maximum token length for Unigram (default: 16)
            shrinking_factor: Shrinking factor for Unigram vocabulary pruning (default: 0.75)
            n_sub_iterations: Number of EM sub-iterations for Unigram (default: 2)
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

        print(f"Training {algorithm.upper()} tokenizer with vocab_size={vocab_size}")

        # Initialize tokenizer based on algorithm
        if algorithm == "bpe":
            tokenizer = Tokenizer(models.BPE())
        elif algorithm == "unigram":
            tokenizer = Tokenizer(models.Unigram())
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Set pre-tokenizer based on strategy
        if pre_tokenizer is None or pre_tokenizer == "byte_level":
            # Simple ByteLevel pretokenization
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
            print("Using ByteLevel pretokenization")

        elif pre_tokenizer == "apertus":
            # Apertus style: regex split + byte-level encoding
            # Pattern from Apertus tokenizer
            apertus_pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(
                    pattern=apertus_pattern,
                    behavior="isolated",
                    invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            ])
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            print("Using Apertus style pretokenization (regex split + ByteLevel)")

        elif pre_tokenizer == "gpt2":
            # GPT-2 style: regex split + byte-level encoding
            gpt2_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(
                    pattern=gpt2_pattern,
                    behavior="isolated",
                    invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            ])
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            print("Using GPT-2 style pretokenization (regex split + ByteLevel)")

        elif pre_tokenizer == "gpt4":
            # GPT-4 style: regex split + byte-level encoding
            gpt4_pattern = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
            tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(
                    pattern=gpt4_pattern,
                    behavior="isolated",
                    invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
            ])
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
            print("Using GPT-4 style pretokenization (regex split + ByteLevel)")

        elif pre_tokenizer == "whitespace":
            # Simple whitespace pretokenization
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.normalizer = normalizers.Sequence([])
            tokenizer.post_processor = None
            print("Using Whitespace pretokenization")

        else:
            raise ValueError(
                f"Unknown pre_tokenizer: {pre_tokenizer}. "
                f"Supported: 'byte_level', 'apertus', 'gpt2', 'gpt4', 'whitespace'"
            )

        # Trainer based on algorithm
        if algorithm == "bpe":
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                show_progress=True,
            )
        elif algorithm == "unigram":
            print(f"Unigram parameters: max_piece_length={max_piece_length}, "
                  f"shrinking_factor={shrinking_factor}, n_sub_iterations={n_sub_iterations}")
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                show_progress=True,
                max_piece_length=max_piece_length,
                shrinking_factor=shrinking_factor,
                n_sub_iterations=n_sub_iterations,
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

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
            "pre_tokenizer": pre_tokenizer or "byte_level",
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

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
        )

        # Save in HF format
        tokenizer.save_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")

        return tokenizer
