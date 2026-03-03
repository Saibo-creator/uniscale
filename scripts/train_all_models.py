"""
Train all language models based on YAML configuration.

Supports both epoch-based and scaling-law (max_steps) training.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

# Allow importing from the project root (src/uniscale/...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.uniscale.models.architectures.model_config import estimate_parameters_for_size


@dataclass
class TrainingHyperparameters:
    """Training hyperparameters."""

    # Either num_train_epochs OR Chinchilla-derived max_steps (via tokens_per_param)
    num_train_epochs: Optional[int] = None
    tokens_per_param: Optional[int] = None

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    tokens_per_step: int = 262144  # gradient_accumulation_steps is derived from this at runtime
    learning_rate: float = 5e-4
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None
    max_length: int = 2048
    bf16: bool = True
    fp16: bool = False
    save_steps: int = 1000
    save_strategy: str = "steps"
    save_total_limit: int = 3
    eval_steps: int = 1000
    evaluation_strategy: str = "steps"
    logging_steps: int = 100
    logging_first_step: bool = True
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""

    enabled: bool = True
    project: str = "scale-invariant-tokenizer"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class ModelTrainingConfig:
    """Configuration for training all models."""

    data_dir: str
    eval_docs: int
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
            data_dir=config_dict["data_dir"],
            eval_docs=config_dict.get("eval_docs", 3000),
            model_sizes=config_dict["model_sizes"],
            tokenizers=config_dict["tokenizers"],
            seeds=config_dict["seeds"],
            training=training_params,
            wandb=wandb_config,
        )


def _get_tokenizer_vocab_size(tokenizer_path: str) -> int:
    """Return the effective vocab size (including added special tokens) for a local tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return len(tok)


def _compute_scaling_law_steps(
    model_size: str,
    tokenizer_path: str,
    tokens_per_param: int,
    actual_tokens_per_step: int,
) -> tuple[int, int, int]:
    """
    Derive max_steps from the Chinchilla tokens-per-parameter ratio.

    Returns (max_steps, total_tokens, param_count).
    """
    vocab_size = _get_tokenizer_vocab_size(tokenizer_path)
    param_count = estimate_parameters_for_size(model_size, vocab_size)
    total_tokens = tokens_per_param * param_count
    max_steps = total_tokens // actual_tokens_per_step
    return max_steps, total_tokens, param_count


def train_model(
    model_size: str,
    tokenizer_path: str,
    seed: int,
    config: ModelTrainingConfig,
    num_gpus: int = 1,
    output_base_dir: str = "out/models",
) -> bool:
    """
    Train a single model.

    Args:
        model_size: Size of model (e.g., "50M")
        tokenizer_path: Path to tokenizer
        seed: Random seed
        config: Full configuration
        num_gpus: Number of GPUs to use (1 for single GPU, >1 for DDP)
        output_base_dir: Base directory for model outputs

    Returns:
        True if successful, False otherwise
    """
    # Derive gradient_accumulation_steps from tokens_per_step
    tokens_per_device_step = config.training.per_device_train_batch_size * config.training.max_length
    gradient_accumulation_steps = config.training.tokens_per_step // (tokens_per_device_step * num_gpus)
    gradient_accumulation_steps = max(1, gradient_accumulation_steps)
    actual_tokens_per_step = tokens_per_device_step * num_gpus * gradient_accumulation_steps
    if actual_tokens_per_step != config.training.tokens_per_step:
        print(
            f"  Warning: tokens_per_step={config.training.tokens_per_step:,} is not exactly divisible "
            f"by batch×max_length×gpus={tokens_per_device_step * num_gpus:,}. "
            f"Using gradient_accumulation_steps={gradient_accumulation_steps} "
            f"→ actual tokens_per_step={actual_tokens_per_step:,}"
        )
    else:
        print(f"  gradient_accumulation_steps={gradient_accumulation_steps} ({actual_tokens_per_step:,} tokens/step)")

    # Calculate max_steps if using scaling law training
    max_steps = None
    total_tokens = None

    if config.training.tokens_per_param is not None:
        max_steps, total_tokens, param_count = _compute_scaling_law_steps(
            model_size, tokenizer_path, config.training.tokens_per_param, actual_tokens_per_step
        )
        print(
            f"  Model params: {param_count/1e9:.3f}B  →  "
            f"target tokens: {total_tokens/1e9:.2f}B  →  "
            f"max_steps: {max_steps:,}  (tokens_per_param={config.training.tokens_per_param})"
        )

    # Build command - use torchrun for DDP if num_gpus > 1
    if num_gpus > 1:
        cmd = [
            sys.executable,  # Use current Python interpreter
            "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "src/uniscale/models/train_lm.py",
        ]
    else:
        cmd = [
            sys.executable,  # Use current Python interpreter
            "src/uniscale/models/train_lm.py",
        ]

    # Construct output directory path
    tokenizer_name = Path(tokenizer_path).name
    output_dir = f"{output_base_dir}/{model_size}_{tokenizer_name}_seed{seed}"

    # Add training arguments
    cmd.extend([
        "--model_size", model_size,
        "--tokenizer_path", tokenizer_path,
        "--train_dir", config.data_dir,
        "--eval_docs", str(config.eval_docs),
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--per_device_train_batch_size", str(config.training.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(config.training.learning_rate),
        "--max_length", str(config.training.max_length),
        "--save_steps", str(config.training.save_steps),
        "--logging_steps", str(config.training.logging_steps),
        "--wandb_project", config.wandb.project,
    ])

    # Add either num_train_epochs OR max_steps (not both)
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    elif config.training.num_train_epochs is not None:
        cmd.extend(["--num_train_epochs", str(config.training.num_train_epochs)])
    else:
        raise ValueError("Must specify either num_train_epochs or tokens_per_param in config")

    # Add eval settings if eval split is requested
    if config.eval_docs > 0:
        cmd.extend(["--eval_steps", str(config.training.eval_steps)])

    # Add warmup configuration
    if config.training.warmup_steps is not None:
        cmd.extend(["--warmup_steps", str(config.training.warmup_steps)])
    elif config.training.warmup_ratio is not None and max_steps is not None:
        # Calculate warmup steps from ratio
        warmup_steps = int(max_steps * config.training.warmup_ratio)
        cmd.extend(["--warmup_steps", str(warmup_steps)])

    # Add precision flags
    if config.training.bf16:
        cmd.append("--bf16")
    if config.training.fp16:
        cmd.append("--fp16")

    # Run training
    print(f"\n[COMMAND] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Training interrupted by user")
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
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (1 for single GPU, >1 for DDP with torchrun)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/models",
        help="Base output directory for model checkpoints (default: out/models)",
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

    # Determine training mode
    using_scaling_law = config.training.tokens_per_param is not None

    print(f"\n{'='*70}")
    print("Training Configuration")
    print(f"{'='*70}")
    print(f"Config file: {args.config}")
    print(f"Training mode: {'Scaling Law (tokens_per_param)' if using_scaling_law else 'Epoch-based'}")
    print(f"GPUs: {args.num_gpus} ({'DDP with torchrun' if args.num_gpus > 1 else 'Single GPU'})")
    print(f"Model sizes: {model_sizes}")

    if using_scaling_law:
        num_gpus = args.num_gpus

        _tpds = config.training.per_device_train_batch_size * config.training.max_length
        _grad_accum = max(1, config.training.tokens_per_step // (_tpds * num_gpus))
        _actual_tps = _tpds * num_gpus * _grad_accum

        print(f"\nScaling law targets ({config.training.tokens_per_param} tokens/param) - Using {num_gpus} GPU(s):")
        for size in model_sizes:
            for tok_path in tokenizers:
                steps, total_tokens, param_count = _compute_scaling_law_steps(
                    size, tok_path, config.training.tokens_per_param, _actual_tps
                )
                tok_name = Path(tok_path).name
                print(
                    f"  {size:>5s} + {tok_name}: {param_count/1e9:.3f}B params → "
                    f"{total_tokens/1e9:.2f}B tokens → {steps:,} steps"
                )

    print(f"\nTokenizers ({len(tokenizers)}):")
    for tok in tokenizers:
        print(f"  - {Path(tok).name}")

    print(f"\nSeeds: {seeds}")
    print(f"Total training runs: {total_runs}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[DRY RUN MODE] - Commands will not be executed\n")
    else:
        response = input(f"Start {total_runs} training run(s)? [y/N] ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            return

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

                success = train_model(
                    model_size,
                    tokenizer_path,
                    seed,
                    config,
                    num_gpus=args.num_gpus,
                    output_base_dir=args.output_dir,
                )

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
