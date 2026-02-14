"""
Train tokenizers (BPE, UnigramLM) with different vocabulary sizes.

This script ensures consistent pretokenization across all tokenizer types
for fair comparison.
"""

import argparse
import json
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
)
from tqdm import tqdm
import sentencepiece as spm


def align_vocab_size_to_256(vocab_size: int) -> int:
    """
    Align vocabulary size to the nearest multiple of 256 for CUDA efficiency.

    CUDA operations are optimized for memory aligned to 256 bytes.
    Aligning vocab size to 256 multiples can provide 5-15% performance improvement.

    Args:
        vocab_size: Original vocabulary size

    Returns:
        Aligned vocabulary size (nearest multiple of 256)
    """
    remainder = vocab_size % 256
    if remainder == 0:
        return vocab_size

    # Round to nearest (round up if exactly at midpoint or beyond)
    if remainder < 128:
        # Round down
        aligned = vocab_size - remainder
    else:
        # Round up (including remainder == 128)
        aligned = vocab_size + (256 - remainder)

    return aligned


def get_text_iterator(data_file: str, batch_size: int = 1000) -> Iterator[List[str]]:
    """
    Iterator to yield batches of text from JSONL file.

    Args:
        data_file: Path to JSONL file with 'text' field
        batch_size: Number of texts per batch

    Yields:
        Batches of text strings
    """
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


def train_bpe_tokenizer(
    data_file: str,
    vocab_size: int,
    output_dir: str,
    min_frequency: int = 2,
) -> None:
    """
    Train a Byte-Level BPE tokenizer.

    Args:
        data_file: Path to training data (JSONL)
        vocab_size: Size of vocabulary (will be aligned to nearest 256 multiple)
        output_dir: Directory to save trained tokenizer
        min_frequency: Minimum frequency for tokens
    """
    # Align vocab size to 256 for CUDA efficiency
    original_vocab_size = vocab_size
    vocab_size = align_vocab_size_to_256(vocab_size)

    if vocab_size != original_vocab_size:
        print(f"Aligning vocab size: {original_vocab_size:,} → {vocab_size:,} (nearest 256 multiple)")

    print(f"Training BPE tokenizer with vocab_size={vocab_size:,}")

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
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
        show_progress=True,
    )

    # Train on iterator
    print("Reading training data...")
    tokenizer.train_from_iterator(
        get_text_iterator(data_file),
        trainer=trainer,
        length=None,  # Will show progress without knowing total
    )

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path / "tokenizer.json"))

    print(f"✓ Saved BPE tokenizer to {output_path}")

    # Save config
    config = {
        "tokenizer_type": "BPE",
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def train_unigram_tokenizer(
    data_file: str,
    vocab_size: int,
    output_dir: str,
) -> None:
    """
    Train a UnigramLM tokenizer using SentencePiece.

    Args:
        data_file: Path to training data (JSONL)
        vocab_size: Size of vocabulary (will be aligned to nearest 256 multiple)
        output_dir: Directory to save trained tokenizer
    """
    # Align vocab size to 256 for CUDA efficiency
    original_vocab_size = vocab_size
    vocab_size = align_vocab_size_to_256(vocab_size)

    if vocab_size != original_vocab_size:
        print(f"Aligning vocab size: {original_vocab_size:,} → {vocab_size:,} (nearest 256 multiple)")

    print(f"Training UnigramLM tokenizer with vocab_size={vocab_size:,}")

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
        model_type="unigram",
        character_coverage=1.0,  # For byte-level
        byte_fallback=True,  # Byte-level fallback
        split_digits=True,
        split_by_unicode_script=False,
        normalization_rule_name="identity",  # No normalization for consistency
        add_dummy_prefix=False,  # No prefix space
        user_defined_symbols=["<mask>"],  # Only custom tokens, control tokens defined via *_id
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

    # Clean up temp file
    temp_text_file.unlink()

    print(f"✓ Saved SentencePiece model to {output_path}")
    print(f"  - tokenizer.model (HF-compatible name)")
    print(f"  - tokenizer.vocab")

    # Create HuggingFace-compatible tokenizer_config.json
    # This allows AutoTokenizer to automatically load the SentencePiece model
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
        "tokenizer_class": "LlamaTokenizer",  # Use Llama tokenizer class for SPM
        "vocab_size": vocab_size,
    }

    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Also save our custom config
    config = {
        "tokenizer_type": "UnigramLM",
        "vocab_size": vocab_size,
        "backend": "sentencepiece",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ Created HuggingFace-compatible config")
    print(f"  Can now load with: AutoTokenizer.from_pretrained('{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Train tokenizers for language model experiments"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["bpe", "unigram"],
        help="Tokenization algorithm to use",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        choices=[80000, 128000, 256000],
        help="Vocabulary size",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/raw/tokenizer_data.jsonl",
        help="Path to training data (JSONL format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: tokenizers/trained/{algorithm}_v{vocab_size})",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens (BPE only)",
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        vocab_size_k = args.vocab_size // 1000
        output_dir = f"tokenizers/trained/{args.algorithm}_v{vocab_size_k}k"
    else:
        output_dir = args.output_dir

    print(f"\n{'='*60}")
    print(f"Training {args.algorithm.upper()} tokenizer")
    print(f"Vocabulary size: {args.vocab_size:,}")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Train tokenizer based on algorithm
    if args.algorithm == "bpe":
        train_bpe_tokenizer(
            data_file=args.data_file,
            vocab_size=args.vocab_size,
            output_dir=output_dir,
            min_frequency=args.min_frequency,
        )
    elif args.algorithm == "unigram":
        train_unigram_tokenizer(
            data_file=args.data_file,
            vocab_size=args.vocab_size,
            output_dir=output_dir,
        )

    print(f"\n✓ Tokenizer training completed!")


if __name__ == "__main__":
    main()
