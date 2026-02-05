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
from datasets import load_dataset
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
    # Check if it's a SentencePiece model or HF tokenizer
    tokenizer_dir = Path(tokenizer_path)

    if (tokenizer_dir / "sentencepiece.model").exists():
        # SentencePiece/Unigram tokenizer
        from transformers import LlamaTokenizer

        tokenizer = LlamaTokenizer(
            vocab_file=str(tokenizer_dir / "sentencepiece.model"),
            legacy=False,
        )
        # Set special tokens
        tokenizer.pad_token = "<pad>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.unk_token = "<unk>"

    elif (tokenizer_dir / "tokenizer.json").exists():
        # HF tokenizers (BPE)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = "<pad>"

    else:
        raise ValueError(f"No valid tokenizer found in {tokenizer_path}")

    return tokenizer


def prepare_dataset(
    data_file: str,
    tokenizer,
    max_length: int = 2048,
    num_proc: int = 4,
):
    """
    Prepare dataset for training.

    Args:
        data_file: Path to training data (JSONL)
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        num_proc: Number of processes for preprocessing

    Returns:
        Tokenized dataset
    """
    # Load dataset
    dataset = load_dataset("json", data_files=data_file, split="train")

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

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description="Train language model with HF Trainer")

    # Model arguments
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=["50M", "100M", "300M", "1B", "3B"],
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
        "--train_file",
        type=str,
        default="data/raw/train_data.jsonl",
        help="Training data file (JSONL)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Evaluation data file (JSONL)",
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
        default=3,
        help="Number of training epochs",
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

    # Set seed
    set_seed(args.seed)

    # Determine output directory
    if args.output_dir is None:
        tokenizer_name = Path(args.tokenizer_path).name
        args.output_dir = f"out/models/{args.model_size}_{tokenizer_name}_seed{args.seed}"

    # Determine run name
    if args.run_name is None:
        tokenizer_name = Path(args.tokenizer_path).name
        args.run_name = f"{args.model_size}_{tokenizer_name}_seed{args.seed}"

    print(f"\n{'='*60}")
    print(f"Training {args.model_size} model")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer_path)
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Get model config
    print(f"Creating {args.model_size} model config...")
    config = get_apertus_config(args.model_size, vocab_size=len(tokenizer))
    estimated_params = estimate_parameters(config)
    print(f"Estimated parameters: {estimated_params:,} (~{estimated_params/1e6:.1f}M)")

    # Initialize model
    print("Initializing model...")
    model = AutoModelForCausalLM.from_config(config)
    print(f"Model initialized with {model.num_parameters():,} parameters")

    # Prepare dataset
    print("Preparing training dataset...")
    train_dataset = prepare_dataset(
        args.train_file,
        tokenizer,
        max_length=args.max_length,
    )
    print(f"Training dataset: {len(train_dataset)} examples")

    eval_dataset = None
    if args.eval_file:
        print("Preparing evaluation dataset...")
        eval_dataset = prepare_dataset(
            args.eval_file,
            tokenizer,
            max_length=args.max_length,
        )
        print(f"Evaluation dataset: {len(eval_dataset)} examples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        bf16=args.bf16,
        fp16=not args.bf16 and torch.cuda.is_available(),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=args.run_name,
        seed=args.seed,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
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
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
    }

    with open(Path(args.output_dir) / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✓ Training completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
