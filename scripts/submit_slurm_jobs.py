#!/usr/bin/env python3
"""
Submit SLURM jobs for distributed training.

This script reads the training configuration and submits SLURM jobs
for each combination of model size, tokenizer, and seed.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional

import yaml


def submit_slurm_job(
    model_size: str,
    tokenizer_path: str,
    seed: int,
    config_file: str,
    output_dir: str,
    nodes: int = 4,
    gpus_per_node: int = 4,
    time: str = "48:00:00",
    partition: str = "normal",
    account: str = None,
    job_name: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    Submit a SLURM job for training.

    Args:
        model_size: Model size (e.g., "50M")
        tokenizer_path: Path to tokenizer
        seed: Random seed
        config_file: Path to YAML config file
        output_dir: Base output directory
        nodes: Number of nodes to use
        gpus_per_node: GPUs per node
        time: Time limit (HH:MM:SS)
        partition: SLURM partition name
        account: SLURM account/project name
        job_name: Job name (optional)
        dry_run: If True, print command without submitting

    Returns:
        True if successful, False otherwise
    """
    tokenizer_name = Path(tokenizer_path).name

    if job_name is None:
        job_name = f"{model_size}_{tokenizer_name}_s{seed}"

    # Build environment variables to pass to SLURM
    env_vars = [
        f"MODEL_SIZE={model_size}",
        f"TOKENIZER_PATH={tokenizer_path}",
        f"SEED={seed}",
        f"CONFIG_FILE={config_file}",
        f"OUTPUT_DIR={output_dir}",
    ]
    env_string = ",".join(["ALL"] + env_vars)

    # Build sbatch command
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--nodes={nodes}",
        f"--gpus-per-node={gpus_per_node}",
        f"--time={time}",
        f"--partition={partition}",
        f"--account={account}",
        f"--export={env_string}",
        "scripts/slurm_train.sbatch",
    ]

    print(f"\n{'='*70}")
    print(f"Submitting job: {job_name}")
    print(f"{'='*70}")
    print(f"Model: {model_size}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Seed: {seed}")
    print(f"Resources: {nodes} nodes × {gpus_per_node} GPUs = {nodes * gpus_per_node} GPUs")
    print(f"Time limit: {time}")
    print(f"Account: {account}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("[DRY RUN] Job not submitted\n")
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Job submitted: {result.stdout.strip()}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job: {e.stderr}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs for distributed training"
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
        "--nodes",
        type=int,
        default=4,
        help="Number of nodes per job (default: 4)",
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=4,
        help="GPUs per node (default: 4)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="48:00:00",
        help="Time limit per job (default: 48:00:00)",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="normal",
        help="SLURM partition name (default: gpu)",
    )
    parser.add_argument(
        "--account",
        type=str,
        required=True,
        help="SLURM account/project name (required, e.g., -A <account>)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/models",
        help="Base output directory (default: out/models)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without submitting jobs",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Submit jobs sequentially with dependency (wait for previous job)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Filter configurations
    model_sizes = [args.model_size] if args.model_size else config["model_sizes"]
    tokenizers = [args.tokenizer] if args.tokenizer else config["tokenizers"]
    seeds = [args.seed] if args.seed else config["seeds"]

    total_jobs = len(model_sizes) * len(tokenizers) * len(seeds)
    total_gpus = args.nodes * args.gpus_per_node

    print(f"\n{'='*70}")
    print("SLURM Job Submission Configuration")
    print(f"{'='*70}")
    print(f"Config file: {args.config}")
    print(f"Model sizes: {model_sizes}")
    print(f"Tokenizers: {len(tokenizers)}")
    for tok in tokenizers:
        print(f"  - {Path(tok).name}")
    print(f"Seeds: {seeds}")
    print(f"Total jobs to submit: {total_jobs}")
    print(f"\nResources per job:")
    print(f"  Nodes: {args.nodes}")
    print(f"  GPUs per node: {args.gpus_per_node}")
    print(f"  Total GPUs: {total_gpus}")
    print(f"  Time limit: {args.time}")
    print(f"  Partition: {args.partition}")
    print(f"  Account: {args.account}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[DRY RUN MODE] - Jobs will not be submitted\n")
    else:
        response = input(f"Submit {total_jobs} job(s) to SLURM? [y/N] ")
        if response.lower() not in ["y", "yes"]:
            print("Aborted.")
            return

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Submit jobs
    successful = 0
    failed = 0
    previous_job_id = None

    for model_size in model_sizes:
        for tokenizer_path in tokenizers:
            for seed in seeds:
                # If sequential mode, add dependency on previous job
                if args.sequential and previous_job_id:
                    print(f"  → Dependency: afterok:{previous_job_id}")

                success = submit_slurm_job(
                    model_size=model_size,
                    tokenizer_path=tokenizer_path,
                    seed=seed,
                    config_file=args.config,
                    output_dir=args.output_dir,
                    nodes=args.nodes,
                    gpus_per_node=args.gpus_per_node,
                    time=args.time,
                    partition=args.partition,
                    account=args.account,
                    dry_run=args.dry_run,
                )

                if success:
                    successful += 1
                else:
                    failed += 1

    print(f"\n{'='*70}")
    print("Job Submission Complete!")
    print(f"{'='*70}")
    print(f"Submitted: {successful}/{total_jobs}")
    print(f"Failed: {failed}/{total_jobs}")
    print(f"\nMonitor jobs with: squeue -u $USER")
    print(f"Cancel all jobs: scancel -u $USER")
    print(f"View logs: tail -f logs/slurm-<job_id>.out")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
