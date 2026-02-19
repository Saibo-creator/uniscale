"""
SentencePiece backend.

This backend uses the SentencePiece library for training.
Supports: UnigramLM, BPE (character-level)
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import sentencepiece as spm
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from uniscale.tokenizers.backends.base import TokenizerBackend


class SentencePieceBackend(TokenizerBackend):
    """
    Backend for SentencePiece library.

    Supports:
    - unigram: Unigram Language Model tokenization
    - bpe: Character-level BPE (via SentencePiece)
    """

    def get_supported_algorithms(self) -> List[str]:
        return ["unigram", "bpe"]

    def train(
        self,
        algorithm: str,
        data_file: str,
        vocab_size: int,
        output_dir: str,
        character_coverage: float = 1.0,
        byte_fallback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a tokenizer using SentencePiece library.

        Args:
            algorithm: "unigram" or "bpe"
            data_file: Path to training data (JSONL)
            vocab_size: Target vocabulary size
            output_dir: Directory to save tokenizer
            character_coverage: Character coverage for SentencePiece
            byte_fallback: Enable byte fallback
            **kwargs: Additional parameters

        Returns:
            Dictionary with training metadata
        """
        if algorithm not in self.get_supported_algorithms():
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by SentencePieceBackend. "
                f"Supported: {self.get_supported_algorithms()}"
            )

        print(f"Training {algorithm.upper()} tokenizer with vocab_size={vocab_size}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Extract text to temporary file for SentencePiece
        temp_text_file = output_path / "temp_training_data.txt"
        print("Extracting text for SentencePiece training...")

        with open(temp_text_file, "w", encoding="utf-8") as out_f:
            with open(data_file, "r", encoding="utf-8") as in_f:
                for line in tqdm(in_f, desc="Extracting text"):
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    if text:
                        out_f.write(text + "\n")

        # Train SentencePiece model
        # Use 'tokenizer' as prefix so it creates 'tokenizer.model' (HF compatible name)
        model_prefix = str(output_path / "tokenizer")

        spm.SentencePieceTrainer.train(
            input=str(temp_text_file),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=algorithm,
            character_coverage=character_coverage,
            byte_fallback=byte_fallback,
            split_digits=True,
            split_by_unicode_script=False,
            normalization_rule_name="identity",
            add_dummy_prefix=False,
            user_defined_symbols=["<mask>"],
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
        )

        # Clean up temp file
        temp_text_file.unlink()

        print(f"✓ Saved SentencePiece model to {output_path}")
        print(f"  - tokenizer.model")
        print(f"  - tokenizer.vocab")

        # Save config
        config = {
            "backend": "sentencepiece",
            "algorithm": algorithm,
            "vocab_size": vocab_size,
            "character_coverage": character_coverage,
            "byte_fallback": byte_fallback,
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
        Export SentencePiece model to HuggingFace format.

        Args:
            artifacts_dir: Directory containing tokenizer.model
            output_dir: Directory to save HF tokenizer
            **kwargs: Additional parameters

        Returns:
            Loaded HuggingFace tokenizer
        """
        from transformers import LlamaTokenizer

        artifacts_path = Path(artifacts_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load SentencePiece model
        model_file = artifacts_path / "tokenizer.model"
        if not model_file.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_file}")

        # Create HuggingFace-compatible tokenizer_config.json
        print("Creating HuggingFace-compatible configuration...")

        tokenizer_config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
            "model_max_length": 2048,
            "padding_side": "right",
            "truncation_side": "right",
            "tokenizer_class": "LlamaTokenizer",
        }

        with open(output_path / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Copy model files
        import shutil
        shutil.copy(model_file, output_path / "tokenizer.model")

        vocab_file = artifacts_path / "tokenizer.vocab"
        if vocab_file.exists():
            shutil.copy(vocab_file, output_path / "tokenizer.vocab")

        # Load tokenizer
        tokenizer = LlamaTokenizer.from_pretrained(str(output_path))

        print(f"✓ Exported HuggingFace tokenizer to {output_path}")
        print(f"  Can now load with: AutoTokenizer.from_pretrained('{output_path}')")

        return tokenizer
