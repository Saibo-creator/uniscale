"""
Train all tokenizers based on YAML configuration.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from uniscale.tokenizers.train_tokenizer import train_bpe_tokenizer, train_unigram_tokenizer


@dataclass
class TokenizerConfig:
    """Configuration for a single tokenizer."""

    algorithm: str
    vocab_size: int
    min_frequency: Optional[int] = 2


@dataclass
class TokenizerTrainingConfig:
    """Configuration for training all tokenizers."""

    data_file: str
    tokenizers: List[TokenizerConfig]

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TokenizerTrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        tokenizers = [
            TokenizerConfig(**tok_config) for tok_config in config_dict["tokenizers"]
        ]

        return cls(data_file=config_dict["data_file"], tokenizers=tokenizers)


def train_tokenizer_from_config(
    config: TokenizerConfig, data_file: str, output_base_dir: str = "tokenizers/trained"
) -> str:
    """
    Train a tokenizer based on configuration.

    Args:
        config: TokenizerConfig
        data_file: Path to training data
        output_base_dir: Base directory for output

    Returns:
        Output directory path
    """
    vocab_size_k = config.vocab_size // 1000
    output_dir = f"{output_base_dir}/{config.algorithm}_v{vocab_size_k}k"

    print(f"\n{'='*60}")
    print(f"Training {config.algorithm.upper()} tokenizer")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    if config.algorithm == "bpe":
        train_bpe_tokenizer(
            data_file=data_file,
            vocab_size=config.vocab_size,
            output_dir=output_dir,
            min_frequency=config.min_frequency,
        )
    elif config.algorithm == "unigram":
        train_unigram_tokenizer(
            data_file=data_file,
            vocab_size=config.vocab_size,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train all tokenizers from YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/tokenizer_training.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="out/tokenizers",
        help="Base directory for trained tokenizers",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = TokenizerTrainingConfig.from_yaml(args.config)

    print(f"Data file: {config.data_file}")
    print(f"Number of tokenizers to train: {len(config.tokenizers)}")

    # Train all tokenizers
    trained_paths = []
    for i, tok_config in enumerate(config.tokenizers, 1):
        print(f"\n[{i}/{len(config.tokenizers)}] Training tokenizer...")

        try:
            output_dir = train_tokenizer_from_config(
                tok_config, config.data_file, args.output_base_dir
            )
            trained_paths.append(output_dir)
            print(f"✓ Successfully trained: {output_dir}")

        except Exception as e:
            print(f"✗ Error training tokenizer: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Successfully trained {len(trained_paths)}/{len(config.tokenizers)} tokenizers")
    print(f"\nTrained tokenizers:")
    for path in trained_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
