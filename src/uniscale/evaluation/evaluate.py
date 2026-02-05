"""
Evaluate trained language models on test data.

Computes perplexity per byte and other metrics.
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uniscale.evaluation.metrics import compute_perplexity_per_byte


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """
    Load model and tokenizer from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    model.eval()

    return model, tokenizer


def load_eval_texts(eval_file: str, max_examples: int = None) -> List[str]:
    """
    Load evaluation texts from JSONL file.

    Args:
        eval_file: Path to JSONL file
        max_examples: Maximum number of examples to load

    Returns:
        List of text strings
    """
    texts = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            data = json.loads(line)
            text = data.get("text", "")
            if text:
                texts.append(text)

    return texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate language model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Evaluation data file (JSONL)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Evaluating model: {args.model_path}")
    print(f"Evaluation file: {args.eval_file}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Load evaluation texts
    print(f"Loading evaluation texts from {args.eval_file}")
    texts = load_eval_texts(args.eval_file, args.max_examples)
    print(f"Loaded {len(texts)} texts")

    # Compute metrics
    print("\nComputing metrics...")
    ppl_per_byte, bits_per_byte, ppl_per_token = compute_perplexity_per_byte(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Perplexity per byte:  {ppl_per_byte:.4f}")
    print(f"Bits per byte:        {bits_per_byte:.4f}")
    print(f"Perplexity per token: {ppl_per_token:.4f}")
    print(f"{'='*60}\n")

    # Prepare results
    results = {
        "model_path": args.model_path,
        "eval_file": args.eval_file,
        "num_examples": len(texts),
        "perplexity_per_byte": ppl_per_byte,
        "bits_per_byte": bits_per_byte,
        "perplexity_per_token": ppl_per_token,
    }

    # Load training info if available
    training_info_file = Path(args.model_path) / "training_info.json"
    if training_info_file.exists():
        with open(training_info_file, "r") as f:
            training_info = json.load(f)
            results.update(training_info)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_path}")
    else:
        # Print as JSON
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
