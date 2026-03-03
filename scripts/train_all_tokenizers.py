"""
Train all tokenizers based on YAML configuration using the beta unified backend interface.

This script supports training tokenizers using multiple backends:
- HuggingFace (bpe, unigram, wordpiece)
- SentencePiece (bpe, unigram, char, word)
- Parity-aware BPE (pa-bpe, pa-super-bpe)
- TkTkT (picky, kudopiece, ngram, bpe-dropout, etc.)

All tokenizers are exported to HuggingFace format for consistent usage.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add parent directory to path to import uniscale
sys.path.insert(0, str(Path(__file__).parent.parent))

from uniscale.tokenizers.train_tokenizer import train_tokenizer


@dataclass
class TokenizerConfig:
    """Configuration for a single tokenizer."""

    algorithm: str
    vocab_size: int
    name: Optional[str] = None       # Used as output dir name if set
    backend: Optional[str] = None    # Auto-detected if None
    output_dir: Optional[str] = None  # Auto-generated if None
    export_hf: bool = True

    # Common parameters
    min_frequency: Optional[int] = None
    character_coverage: Optional[float] = None
    num_workers: Optional[int] = None

    # HuggingFace-specific parameters
    pre_tokenizer: Optional[str] = None  # "byte_level", "apertus", "gpt2", "gpt4", or "whitespace"

    # SuperBPE backend parameters (superbpe algorithm)
    corpus_dir: Optional[str] = None             # directory of .txt files (replaces data_file)
    num_bytes: Optional[int] = None              # bytes of training data to use
    pre_tokenizer_phase2: Optional[str] = None   # Phase 2 regex preset (default: "superbpe")
    superbpe_venv: Optional[str] = None          # path to superbpe venv (overrides hardcoded default)
    superbpe_dir: Optional[str] = None           # path to superbpe source dir (overrides hardcoded default)

    # HuggingFace / parity-aware super-BPE parameters
    phase1_merges: Optional[int] = None

    # Parity-aware BPE parameters
    dev_file: Optional[str] = None
    ratio: Optional[List[float]] = None

    # TkTkT-specific parameters
    picky_threshold: Optional[float] = None
    max_type_length: Optional[int] = None
    dropout_probability: Optional[float] = None
    initial_vocab_size: Optional[int] = None
    shrinking_factor: Optional[float] = None
    num_sub_iterations: Optional[int] = None
    ngram_n: Optional[int] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert config to kwargs for train_tokenizer function."""
        kwargs = {}
        for key, value in self.__dict__.items():
            if value is not None and key not in ["algorithm", "vocab_size", "name", "output_dir"]:
                kwargs[key] = value
        return kwargs


@dataclass
class TokenizerTrainingConfig:
    """Configuration for training all tokenizers."""

    data_file: str
    tokenizers: List[TokenizerConfig]
    defaults: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "TokenizerTrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Get default values
        defaults = config_dict.get("defaults", {})

        # Parse tokenizer configs
        tokenizers = []
        for tok_config in config_dict["tokenizers"]:
            # Merge with defaults (tok_config takes precedence)
            merged_config = {**defaults, **tok_config}
            tokenizers.append(TokenizerConfig(**merged_config))

        return cls(
            data_file=config_dict["data_file"],
            tokenizers=tokenizers,
            defaults=defaults,
        )


def train_tokenizer_from_config(
    config: TokenizerConfig,
    data_file: str,
    output_base_dir: str = "tokenizers/trained",
    dry_run: bool = False,
) -> Optional[str]:
    """
    Train a tokenizer based on configuration.

    Args:
        config: TokenizerConfig
        data_file: Path to training data
        output_base_dir: Base directory for output
        dry_run: If True, only print what would be done

    Returns:
        Output directory path, or None if dry_run
    """
    # Generate output directory: explicit > name field > auto-generated
    if config.output_dir is not None:
        output_dir = config.output_dir
    elif config.name is not None:
        output_dir = f"{output_base_dir}/{config.name}"
    else:
        vocab_size_k = config.vocab_size // 1000
        backend_str = f"_{config.backend}" if config.backend else ""
        output_dir = f"{output_base_dir}/{config.algorithm}{backend_str}_v{vocab_size_k}k"

    print(f"\n{'='*60}")
    print(f"Training {config.algorithm.upper()} tokenizer")
    if config.backend:
        print(f"Backend: {config.backend}")
    print(f"Vocabulary size: {config.vocab_size:,}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    if dry_run:
        print("\n[DRY RUN] Would train with parameters:")
        kwargs = config.to_kwargs()
        for key, value in kwargs.items():
            print(f"  {key}: {value}")
        print()
        return None

    # Get kwargs for training
    kwargs = config.to_kwargs()

    # Train tokenizer
    try:
        output_path = train_tokenizer(
            algorithm=config.algorithm,
            data_file=data_file,
            vocab_size=config.vocab_size,
            output_dir=output_dir,
            **kwargs,
        )
        return output_path

    except Exception as e:
        print(f"\n✗ Error training tokenizer: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Train all tokenizers from YAML configuration (beta version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/tokenizer_training_beta.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="out/tokenizers",
        help="Base directory for trained tokenizers",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be done without actually training",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter tokenizers by algorithm name (e.g., 'picky' or 'bpe')",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N tokenizers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train at most N tokenizers",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = TokenizerTrainingConfig.from_yaml(args.config)

    print(f"Data file: {config.data_file}")
    print(f"Total tokenizers in config: {len(config.tokenizers)}")

    # Filter tokenizers if requested
    tokenizers_to_train = config.tokenizers
    if args.filter:
        tokenizers_to_train = [
            tok for tok in tokenizers_to_train if args.filter in tok.algorithm
        ]
        print(f"Filtered to {len(tokenizers_to_train)} tokenizers matching '{args.filter}'")

    # Apply skip and limit
    if args.skip > 0:
        tokenizers_to_train = tokenizers_to_train[args.skip :]
        print(f"Skipping first {args.skip} tokenizers")

    if args.limit is not None:
        tokenizers_to_train = tokenizers_to_train[: args.limit]
        print(f"Limited to {args.limit} tokenizers")

    print(f"\nWill train {len(tokenizers_to_train)} tokenizers")

    if args.dry_run:
        print("\n[DRY RUN MODE] - No actual training will occur\n")

    # Train all tokenizers
    trained_paths = []
    failed_count = 0

    for i, tok_config in enumerate(tokenizers_to_train, 1):
        print(f"\n[{i}/{len(tokenizers_to_train)}] Training tokenizer...")

        output_dir = train_tokenizer_from_config(
            tok_config, config.data_file, args.output_base_dir, args.dry_run
        )

        if output_dir:
            trained_paths.append(output_dir)
            print(f"✓ Successfully trained: {output_dir}")
        elif not args.dry_run:
            failed_count += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"Dry run complete - no tokenizers were actually trained")
    else:
        print(f"Successfully trained: {len(trained_paths)}/{len(tokenizers_to_train)}")
        if failed_count > 0:
            print(f"Failed: {failed_count}")

        if trained_paths:
            print(f"\nTrained tokenizers:")
            for path in trained_paths:
                print(f"  - {path}")


if __name__ == "__main__":
    main()
