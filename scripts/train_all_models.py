"""
Train all language models based on YAML configuration.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters."""

    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    warmup_steps: int = 500
    max_length: int = 2048
    bf16: bool = True
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    project: str = "scale-invariant-tokenizer"
    entity: Optional[str] = None


@dataclass
class ModelTrainingConfig:
    """Configuration for training all models."""

    train_file: str
    eval_file: Optional[str]
    model_sizes: List[str]
    tokenizers: List[str]
    seeds: List[int]
    training: TrainingHyperparameters
    wandb: WandBConfig

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "ModelTrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        training_params = TrainingHyperparameters(**config_dict.get("training", {}))
        wandb_config = WandBConfig(**config_dict.get("wandb", {}))

        return cls(
            train_file=config_dict["train_file"],
            eval_file=config_dict.get("eval_file"),
            model_sizes=config_dict["model_sizes"],
            tokenizers=config_dict["tokenizers"],
            seeds=config_dict["seeds"],
            training=training_params,
            wandb=wandb_config,
        )


def train_model(
    model_size: str,
    tokenizer_path: str,
    seed: int,
    config: ModelTrainingConfig,
) -> bool:
    """
    Train a single model.

    Args:
        model_size: Size of model (e.g., "50M")
        tokenizer_path: Path to tokenizer
        seed: Random seed
        config: Full configuration

    Returns:
        True if successful, False otherwise
    """
    # Build command
    cmd = [
        "python",
        "-m",
        "scale.models.train_lm",
        "--model_size",
        model_size,
        "--tokenizer_path",
        tokenizer_path,
        "--train_file",
        config.train_file,
        "--seed",
        str(seed),
        "--num_train_epochs",
        str(config.training.num_train_epochs),
        "--per_device_train_batch_size",
        str(config.training.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(config.training.gradient_accumulation_steps),
        "--learning_rate",
        str(config.training.learning_rate),
        "--warmup_steps",
        str(config.training.warmup_steps),
        "--max_length",
        str(config.training.max_length),
        "--save_steps",
        str(config.training.save_steps),
        "--eval_steps",
        str(config.training.eval_steps),
        "--logging_steps",
        str(config.training.logging_steps),
        "--wandb_project",
        config.wandb.project,
    ]

    if config.eval_file:
        cmd.extend(["--eval_file", config.eval_file])

    if config.training.bf16:
        cmd.append("--bf16")

    if config.wandb.entity:
        cmd.extend(["--wandb_entity", config.wandb.entity])

    # Run training
    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train all models from YAML configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/model_training.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        help="Train only specific model size (optional)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Train only with specific tokenizer (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Train only with specific seed (optional)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    print(f"Loading configuration from {args.config}")
    config = ModelTrainingConfig.from_yaml(args.config)

    # Filter based on command-line arguments
    model_sizes = [args.model_size] if args.model_size else config.model_sizes
    tokenizers = [args.tokenizer] if args.tokenizer else config.tokenizers
    seeds = [args.seed] if args.seed else config.seeds

    # Calculate total number of training runs
    total_runs = len(model_sizes) * len(tokenizers) * len(seeds)

    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}")
    print(f"Model sizes: {model_sizes}")
    print(f"Tokenizers: {len(tokenizers)}")
    print(f"Seeds: {seeds}")
    print(f"Total training runs: {total_runs}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("DRY RUN - Commands will not be executed\n")

    # Train all combinations
    successful = 0
    failed = 0

    for i, model_size in enumerate(model_sizes, 1):
        for j, tokenizer_path in enumerate(tokenizers, 1):
            for k, seed in enumerate(seeds, 1):
                run_num = (
                    (i - 1) * len(tokenizers) * len(seeds)
                    + (j - 1) * len(seeds)
                    + k
                )

                print(f"\n{'='*60}")
                print(f"Training Run [{run_num}/{total_runs}]")
                print(f"{'='*60}")
                print(f"Model size: {model_size}")
                print(f"Tokenizer: {tokenizer_path}")
                print(f"Seed: {seed}")
                print(f"{'='*60}\n")

                if args.dry_run:
                    print(
                        f"Would train: {model_size} + {Path(tokenizer_path).name} + seed={seed}"
                    )
                    continue

                success = train_model(model_size, tokenizer_path, seed, config)

                if success:
                    successful += 1
                    print(f"✓ Training completed successfully")
                else:
                    failed += 1
                    print(f"✗ Training failed")

    print(f"\n{'='*60}")
    print("All Training Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{total_runs}")
    print(f"Failed: {failed}/{total_runs}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
