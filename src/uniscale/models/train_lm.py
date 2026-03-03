"""
Train language models using Hugging Face Trainer.

This script trains Apertus-based models of different sizes on tokenized data.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback

from uniscale.models.architectures.model_config import get_apertus_config, estimate_parameters


def load_tokenizer(tokenizer_path: str):
    """
    Load tokenizer from path.

    Args:
        tokenizer_path: Path to trained tokenizer

    Returns:
        Loaded tokenizer
    """
    # All tokenizers should now be loadable via AutoTokenizer
    # (both BPE with tokenizer.json and Unigram with tokenizer.model + tokenizer_config.json)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    return tokenizer


def prepare_dataset(
    data_file: str,
    tokenizer,
    max_length: int = 2048,
    num_proc: Optional[int] = None,
):
    """
    Prepare dataset for training.

    Args:
        data_file: Path to training data (JSONL)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes for preprocessing (None = auto-detect 80% of cores)

    Returns:
        Tokenized dataset
    """
    # Auto-detect number of processes if not specified
    if num_proc is None:
        import multiprocessing as mp
        num_proc = max(1, int(mp.cpu_count() * 0.8))
        print(f"Auto-detected {num_proc} CPU cores for tokenization (80% of {mp.cpu_count()})")

    # Load dataset (uses HF_DATASETS_CACHE environment variable for cache location)
    dataset = load_dataset(
        "json",
        data_files=data_file,
        split="train",
    )

    # Tokenize function
    def tokenize_function(examples):
        # Tokenize texts
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
        )
        return outputs

    # Tokenize dataset with explicit cache
    # HuggingFace will automatically generate cache filenames based on transformation hash
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
        load_from_cache_file=True,  # Explicitly enable cache loading
    )

    return tokenized_dataset


def prepare_dataset_parquet(
    data_dir: str,
    tokenizer,
    max_length: int = 2048,
    eval_docs: int = 0,
):
    """
    Prepare a lazy streaming dataset for training from a directory of parquet files.

    Scans `data_dir` for sub-directories containing parquet files, interleaves
    them, and yields tokenized chunks of exactly `max_length` tokens.  Long
    documents are split into as many full-length chunks as possible; the
    trailing remainder (< max_length tokens) is dropped so every sample in the
    returned dataset has the same length.

    Args:
        data_dir: Root directory whose immediate sub-directories each contain
                  one or more ``*.parquet`` files.
        tokenizer: Tokenizer to use.
        max_length: Number of tokens per training sample.
        eval_docs: Number of raw documents to hold out as the eval split.
                   The first ``eval_docs`` documents (after interleaving) go to
                   eval; the rest go to train.  Pass 0 to skip the eval split.

    Returns:
        Tuple ``(train_ds, eval_ds)`` of ``IterableDataset`` objects.
        ``eval_ds`` is ``None`` when ``eval_docs=0``.
        NOTE: streaming datasets have no ``len()``.  The Trainer must be
        configured with ``max_steps`` rather than ``num_train_epochs``.
    """
    base_path = Path(data_dir)

    all_streaming_datasets = []
    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue
        if not list(folder.glob("**/*.parquet")):
            continue
        try:
            ds = load_dataset(
                "parquet",
                data_files=str(folder / "**/*.parquet"),
                split="train",
                streaming=True,
            )
            # Normalise to a single "text" column regardless of source schema
            src_columns = list(ds.features.keys())

            def _unify(example, _cols=src_columns):  # default arg captures value, not name
                text = ""
                for key in ("text", "content"):
                    if key in _cols and example.get(key):
                        text = str(example[key])
                        break
                return {"text": text}

            ds = ds.map(_unify, remove_columns=src_columns)
            all_streaming_datasets.append(ds)
            print(f"Loaded parquet dir: {folder.name}")
        except Exception as exc:
            print(f"Skipping {folder.name}: {exc}")

    if not all_streaming_datasets:
        raise ValueError(f"No parquet datasets found under {data_dir}")

    combined = interleave_datasets(all_streaming_datasets)

    def _tokenize_and_chunk(examples):
        # Tokenise full texts without any truncation
        token_lists = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]

        # Split each document into non-overlapping max_length chunks;
        # drop the trailing incomplete chunk so every sample is full-length.
        chunks = []
        for ids in token_lists:
            for start in range(0, len(ids) - max_length + 1, max_length):
                chunks.append(ids[start : start + max_length])
        return {"input_ids": chunks}

    def _to_tokenized(raw_ds):
        return raw_ds.map(_tokenize_and_chunk, batched=True, remove_columns=["text"])

    if eval_docs > 0:
        eval_ds = _to_tokenized(combined.take(eval_docs))
        train_ds = _to_tokenized(combined.skip(eval_docs))
    else:
        train_ds = _to_tokenized(combined)
        eval_ds = None

    return train_ds, eval_ds


def main():
    parser = argparse.ArgumentParser(description="Train language model with HF Trainer")

    # Model arguments
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["30M", "100M", "300M", "1B", "3B", "8B"],
        help="Model size",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to trained tokenizer",
    )

    # Data arguments
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/tokenizer_training_dataset",
        help="Root directory of parquet sub-directories for training",
    )
    parser.add_argument(
        "--eval_docs",
        type=int,
        default=3000,
        help=(
            "Number of raw documents to hold out as eval split "
            "(first N docs from the interleaved stream; 0 = no eval). "
            "After chunking, expect roughly 40-60%% of these docs to yield "
            "usable samples, so 3000 docs ≈ 1200-1800 eval chunks."
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs (use this OR max_steps, not both)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (for scaling law training)",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per device for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="Warmup steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 training",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 training",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="scale-invariant-tokenizer",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name",
    )

    args = parser.parse_args()

    # Validate training config
    if args.num_train_epochs is None and args.max_steps is None:
        # Default to 3 epochs if neither specified
        args.num_train_epochs = 3
    elif args.num_train_epochs is not None and args.max_steps is not None:
        raise ValueError("Specify either num_train_epochs OR max_steps, not both")

    # Get local rank for DDP (rank 0 is the main process)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Debug: Always print this to see if script is running
    print(f"[DEBUG] LOCAL_RANK={os.environ.get('LOCAL_RANK', 'NOT_SET')}, local_rank={local_rank}, is_main_process={is_main_process}", flush=True)

    # Set seed
    set_seed(args.seed)

    # Set W&B project name and configure for DDP
    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Only report to wandb from main process in DDP
    if not is_main_process:
        os.environ["WANDB_DISABLED"] = "true"

    # Determine output directory
    if args.output_dir is None:
        tokenizer_name = Path(args.tokenizer_path).name
        args.output_dir = f"out/models/{args.model_size}_{tokenizer_name}_seed{args.seed}"

    # Determine run name
    if args.run_name is None:
        tokenizer_name = Path(args.tokenizer_path).name
        args.run_name = f"{args.model_size}_{tokenizer_name}_seed{args.seed}"

    # Calculate total tokens if using max_steps
    total_tokens = None
    if args.max_steps is not None:
        tokens_per_step = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.max_length
        )
        total_tokens = args.max_steps * tokens_per_step

    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training {args.model_size} model")
        print(f"Tokenizer: {args.tokenizer_path}")
        print(f"Output: {args.output_dir}")
        print(f"Seed: {args.seed}")
        if args.max_steps:
            print(f"Max steps: {args.max_steps:,}")
            print(f"Total tokens: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
        else:
            print(f"Epochs: {args.num_train_epochs}")
        print(f"{'='*60}\n")

    # Load tokenizer
    if is_main_process:
        print("Loading tokenizer...", flush=True)
    tokenizer = load_tokenizer(args.tokenizer_path)
    if is_main_process:
        print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}", flush=True)

    # Get model config
    if is_main_process:
        print(f"Creating {args.model_size} model config...", flush=True)
        print(f"[DEBUG] About to load Apertus config from HuggingFace Hub...", flush=True)
    config = get_apertus_config(args.model_size, vocab_size=len(tokenizer))
    if is_main_process:
        print(f"[DEBUG] Apertus config loaded successfully", flush=True)
    estimated_params = estimate_parameters(config)
    if is_main_process:
        print(f"Estimated parameters: {estimated_params:,} (~{estimated_params/1e6:.1f}M)")

    # Initialize model
    if is_main_process:
        print("Initializing model...")
    # Determine dtype based on training arguments
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    if is_main_process:
        print(f"Model initialized with {model.num_parameters():,} parameters (dtype: {model.dtype})")

        # Log training precision mode
        if args.fp16:
            print("Will use FP16 automatic mixed precision training")
        elif args.bf16:
            print("Will use BF16 automatic mixed precision training")
        else:
            print("Will use FP32 full precision training")

    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")

    # Streaming datasets read directly from parquet — no disk cache is written,
    # so every DDP rank builds its own stream independently (no barrier needed).
    if is_main_process:
        print(f"Preparing streaming dataset from {args.train_dir} ...", flush=True)
        if not os.path.isdir(args.train_dir):
            raise FileNotFoundError(f"Training directory not found: {args.train_dir}")

    train_dataset, eval_dataset = prepare_dataset_parquet(
        args.train_dir,
        tokenizer,
        max_length=args.max_length,
        eval_docs=args.eval_docs,
    )

    if is_main_process:
        print(f"Streaming dataset ready (eval_docs={args.eval_docs})", flush=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "dataloader_num_workers": 1,  # streaming IterableDataset: num_workers capped at num_shards anyway
        "remove_unused_columns": False,
        "report_to": "wandb",
        "run_name": args.run_name,
        "seed": args.seed,
    }

    # Add either max_steps or num_train_epochs
    if args.max_steps is not None:
        training_args_dict["max_steps"] = args.max_steps
    else:
        training_args_dict["num_train_epochs"] = args.num_train_epochs

    # Add evaluation settings if eval dataset exists
    if eval_dataset:
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["metric_for_best_model"] = "eval_loss"
    else:
        training_args_dict["eval_strategy"] = "no"

    training_args = TrainingArguments(**training_args_dict)


    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    if is_main_process:
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")

    trainer.train()

    # Save final model (only main process saves)
    if is_main_process:
        print(f"\nSaving final model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Save training info
        info = {
            "model_size": args.model_size,
            "tokenizer_path": args.tokenizer_path,
            "vocab_size": len(tokenizer),
            "num_parameters": model.num_parameters(),
            "seed": args.seed,
            "max_length": args.max_length,
            "num_train_epochs": args.num_train_epochs if args.max_steps is None else None,
            "max_steps": args.max_steps if args.max_steps is not None else None,
            "total_tokens": total_tokens if total_tokens is not None else None,
            "learning_rate": args.learning_rate,
        }

        with open(Path(args.output_dir) / "training_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Training completed!")
        print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
