#!/usr/bin/env python3
"""
Load a trained SuperBPE tokenizer and segment sample sentences.

Usage:
    python scripts/load_superbpe.py
    python scripts/load_superbpe.py --tokenizer_dir out/tokenizers/superbpe_v128k
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer

SAMPLES = [
    "Hello, world! This is a test sentence.",
    "The quick brown fox jumps over the lazy dog.",
    "Bonjour le monde! Comment ça va?",
    "日本語のテキストをトークン化します。",
    "สวัสดีครับ นี่คือข้อความภาษาไทย",
    "এটি একটি বাংলা বাক্য।",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


def segment(tokenizer: Tokenizer, text: str) -> None:
    encoding = tokenizer.encode(text)
    tokens = encoding.tokens
    print(f"  Input : {text}")
    print(f"  Tokens: {tokens}")
    print(f"  Count : {len(tokens)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Load SuperBPE tokenizer and segment sentences")
    parser.add_argument(
        "--tokenizer_dir",
        default="out/tokenizers/superbpe_v128k",
        help="Directory containing tokenizer.json",
    )
    args = parser.parse_args()

    tokenizer_file = Path(args.tokenizer_dir) / "tokenizer.json"
    if not tokenizer_file.exists():
        print(f"Error: {tokenizer_file} not found")
        return 1

    print(f"Loading tokenizer from {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    print(f"Vocab size: {tokenizer.get_vocab_size():,}\n")

    for text in SAMPLES:
        segment(tokenizer, text)


if __name__ == "__main__":
    main()
