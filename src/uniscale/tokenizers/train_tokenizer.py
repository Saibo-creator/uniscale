"""
Train tokenizers with different backends and algorithms.

This script provides a unified interface for training tokenizers using different
backend libraries (HuggingFace, SentencePiece, parity-aware-bpe, TkTkT).
All tokenizers are exported to HuggingFace format for consistent usage.
"""

import argparse
from typing import Dict, List

from uniscale.tokenizers.backends import (
    TokenizerBackend,
    HuggingFaceBackend,
    SentencePieceBackend,
    ParityAwareBPEBackend,
    TkTkTBackend,
    SuperBPEBackend,
)


# Registry of available backends
BACKEND_REGISTRY: Dict[str, TokenizerBackend] = {
    "huggingface": HuggingFaceBackend(),
    "sentencepiece": SentencePieceBackend(),
    "parity-aware-bpe": ParityAwareBPEBackend(),
    "tktkt": TkTkTBackend(),
    "superbpe": SuperBPEBackend(),
}


def get_available_algorithms() -> Dict[str, list]:
    """Get all available algorithms grouped by backend."""
    return {
        backend_name: backend.get_supported_algorithms()
        for backend_name, backend in BACKEND_REGISTRY.items()
    }


def find_backend_for_algorithm(algorithm: str) -> tuple[str, TokenizerBackend]:
    """
    Find the backend that supports the given algorithm.

    Args:
        algorithm: The tokenization algorithm

    Returns:
        Tuple of (backend_name, backend_instance)

    Raises:
        ValueError: If no backend supports the algorithm
    """
    for backend_name, backend in BACKEND_REGISTRY.items():
        if algorithm in backend.get_supported_algorithms():
            return backend_name, backend

    # If not found, show available algorithms
    available = get_available_algorithms()
    available_str = "\n".join(
        f"  {backend}: {', '.join(algos)}"
        for backend, algos in available.items()
    )
    raise ValueError(
        f"Algorithm '{algorithm}' not supported by any backend.\n"
        f"Available algorithms:\n{available_str}"
    )


def train_tokenizer(
    algorithm: str,
    data_file: str,
    vocab_size: int,
    output_dir: str = None,
    backend: str = None,
    export_hf: bool = True,
    **kwargs
) -> str:
    """
    Train a tokenizer using the specified algorithm.

    Args:
        algorithm: Tokenization algorithm to use
        data_file: Path to training data (JSONL format)
        vocab_size: Target vocabulary size
        output_dir: Output directory (auto-generated if None)
        backend: Backend to use (auto-detected if None)
        export_hf: Whether to export to HuggingFace format
        **kwargs: Additional algorithm-specific parameters

    Returns:
        Path to the output directory
    """
    # Auto-detect backend if not specified
    if backend is None:
        backend_name, backend_instance = find_backend_for_algorithm(algorithm)
        print(f"Auto-detected backend: {backend_name}")
    else:
        if backend not in BACKEND_REGISTRY:
            raise ValueError(
                f"Backend '{backend}' not found. "
                f"Available backends: {list(BACKEND_REGISTRY.keys())}"
            )
        backend_instance = BACKEND_REGISTRY[backend]
        backend_name = backend

        # Verify algorithm is supported
        if algorithm not in backend_instance.get_supported_algorithms():
            raise ValueError(
                f"Algorithm '{algorithm}' not supported by backend '{backend}'. "
                f"Supported algorithms: {backend_instance.get_supported_algorithms()}"
            )

    # Determine output directory
    if output_dir is None:
        vocab_size_k = vocab_size // 1000
        output_dir = f"out/tokenizers/{algorithm}_v{vocab_size_k}k"

    print(f"\n{'='*60}")
    print(f"Training Tokenizer")
    print(f"{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"Algorithm: {algorithm}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Data file: {data_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Train tokenizer
    artifacts_dir = output_dir
    _ = backend_instance.train(
        algorithm=algorithm,
        data_file=data_file,
        vocab_size=vocab_size,
        output_dir=artifacts_dir,
        **kwargs
    )

    print(f"\n✓ Training completed!")
    print(f"  Artifacts saved to: {artifacts_dir}")

    # Export to HuggingFace format
    if export_hf:
        hf_dir = f"{output_dir}/hf"
        print(f"\nExporting to HuggingFace format...")
        _ = backend_instance.export_to_hf(
            artifacts_dir=artifacts_dir,
            output_dir=hf_dir,
        )
        print(f"✓ HuggingFace tokenizer saved to: {hf_dir}")
        print(f"  Load with: AutoTokenizer.from_pretrained('{hf_dir}')")

    print(f"\n{'='*60}")
    print(f"✓ Tokenizer training completed successfully!")
    print(f"{'='*60}\n")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train tokenizers using different backends and algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        help="Tokenization algorithm to use (e.g., bpe, unigram, parity-bpe)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        required=True,
        help="Target vocabulary size",
    )

    # Optional arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/raw/tokenizer_data.jsonl",
        help="Path to training data (JSONL format with 'text' field)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: out/tokenizers/{algorithm}_v{vocab_size}k)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=list(BACKEND_REGISTRY.keys()),
        help="Backend to use (auto-detected from algorithm if not specified)",
    )
    parser.add_argument(
        "--no_export_hf",
        action="store_true",
        help="Skip exporting to HuggingFace format",
    )

    # Algorithm-specific arguments
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for tokens (BPE)",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=1.0,
        help="Character coverage (SentencePiece, TkTkT)",
    )
    parser.add_argument(
        "--phase1_merges",
        type=int,
        default=None,
        help="For super-bpe variants: number of merges in phase1 (default: 60%% of vocab_size)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel processing (default: 1)",
    )

    # Parity-aware BPE specific arguments (multi-lingual)
    parser.add_argument(
        "--dev_file",
        type=str,
        default=None,
        help="Development data JSONL with language field (alternative to ratio, for pa-bpe, pa-super-bpe)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs='*',
        default=None,
        help="Desired compression ratio per language (alternative to dev_file, for pa-bpe, pa-super-bpe)",
    )

    # TkTkT-specific arguments
    parser.add_argument(
        "--picky_threshold",
        type=float,
        default=0.5,
        help="Threshold for PickyBPE (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--max_type_length",
        type=int,
        default=16,
        help="Maximum token length for TkTkT algorithms (default: 16)",
    )
    parser.add_argument(
        "--dropout_probability",
        type=float,
        default=0.1,
        help="Dropout probability for BPE-dropout (default: 0.1)",
    )
    parser.add_argument(
        "--initial_vocab_size",
        type=int,
        default=1_000_000,
        help="Initial vocab size for KudoPiece (default: 1,000,000)",
    )
    parser.add_argument(
        "--shrinking_factor",
        type=float,
        default=0.75,
        help="Shrinking factor for KudoPiece (default: 0.75)",
    )
    parser.add_argument(
        "--num_sub_iterations",
        type=int,
        default=2,
        help="Number of sub-iterations for KudoPiece (default: 2)",
    )
    parser.add_argument(
        "--ngram_n",
        type=int,
        default=3,
        help="N for N-gram tokenizer (default: 3)",
    )

    args = parser.parse_args()

    # Show available algorithms if requested
    if args.algorithm == "list":
        print("\nAvailable algorithms by backend:\n")
        available = get_available_algorithms()
        for backend_name, algos in available.items():
            print(f"  {backend_name}:")
            for algo in algos:
                print(f"    - {algo}")
        print()
        return

    # Train tokenizer
    train_tokenizer(
        algorithm=args.algorithm,
        data_file=args.data_file,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        backend=args.backend,
        export_hf=not args.no_export_hf,
        min_frequency=args.min_frequency,
        character_coverage=args.character_coverage,
        phase1_merges=args.phase1_merges,
        num_workers=args.num_workers,
        # Parity-aware BPE parameters
        dev_file=args.dev_file,
        ratio=args.ratio,
        # TkTkT-specific parameters
        picky_threshold=args.picky_threshold,
        max_type_length=args.max_type_length,
        dropout_probability=args.dropout_probability,
        initial_vocab_size=args.initial_vocab_size,
        shrinking_factor=args.shrinking_factor,
        num_sub_iterations=args.num_sub_iterations,
        ngram_n=args.ngram_n,
    )


if __name__ == "__main__":
    main()
