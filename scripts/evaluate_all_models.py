"""
Evaluate all trained models and save results.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List


def find_all_checkpoints(checkpoints_dir: str) -> List[str]:
    """
    Find all model checkpoints.

    Args:
        checkpoints_dir: Directory containing checkpoints

    Returns:
        List of checkpoint paths
    """
    checkpoints_path = Path(checkpoints_dir)
    checkpoints = []

    for checkpoint_dir in checkpoints_path.glob("*"):
        if checkpoint_dir.is_dir() and (checkpoint_dir / "training_info.json").exists():
            checkpoints.append(str(checkpoint_dir))

    return sorted(checkpoints)


def evaluate_model(
    model_path: str,
    eval_file: str,
    output_dir: str,
    batch_size: int = 8,
    max_examples: int = None,
) -> bool:
    """
    Evaluate a single model.

    Args:
        model_path: Path to model checkpoint
        eval_file: Evaluation data file
        output_dir: Directory for results
        batch_size: Batch size for evaluation
        max_examples: Maximum examples to evaluate

    Returns:
        True if successful
    """
    model_name = Path(model_path).name
    output_file = Path(output_dir) / f"{model_name}_results.json"

    print(f"\nEvaluating: {model_name}")
    print(f"Output: {output_file}")

    # Build command
    cmd = [
        "python",
        "-m",
        "scale.evaluation.evaluate",
        "--model_path",
        model_path,
        "--eval_file",
        eval_file,
        "--output_file",
        str(output_file),
        "--batch_size",
        str(batch_size),
    ]

    if max_examples:
        cmd.extend(["--max_examples", str(max_examples)])

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Evaluation completed: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")

    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="out/models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Evaluation data file (JSONL)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--model_filter",
        type=str,
        default=None,
        help="Filter models by name pattern (optional)",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all checkpoints
    print(f"Finding checkpoints in {args.checkpoints_dir}")
    checkpoints = find_all_checkpoints(args.checkpoints_dir)

    # Filter if specified
    if args.model_filter:
        checkpoints = [cp for cp in checkpoints if args.model_filter in cp]

    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    if not checkpoints:
        print("No checkpoints found!")
        return

    # Evaluate all models
    successful = 0
    failed = 0

    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(checkpoints)}] Evaluating model")
        print(f"{'='*60}")

        success = evaluate_model(
            model_path=checkpoint,
            eval_file=args.eval_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_examples=args.max_examples,
        )

        if success:
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(checkpoints)}")
    print(f"Failed: {failed}/{len(checkpoints)}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
