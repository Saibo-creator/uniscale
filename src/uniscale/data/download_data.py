"""
Download FineWeb 2 data for specific languages.

This script downloads a subset of FineWeb 2 dataset for tokenizer training
and language model training.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from tqdm import tqdm


# Define language sets
LANG_SETS = {
    "LANG_SET_20": [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja",
        "zh", "ar", "hi", "ko", "tr", "vi", "id", "th", "cs", "ro"
    ],
    "LANG_SET_5": ["en", "es", "fr", "de", "zh"],
    "LANG_SET_1": ["en"],
}


def download_fineweb_subset(
    languages: List[str],
    output_dir: str,
    total_size_gb: float,
    tokenizer_size_gb: float = 2.0,
    split: str = "train",
    num_proc: int = 4,
):
    """
    Download FineWeb 2 data for specified languages.

    Args:
        languages: List of language codes to download
        output_dir: Directory to save the downloaded data
        total_size_gb: Total size of data to download for LM training (in GB)
        tokenizer_size_gb: Size of data for tokenizer training (in GB)
        split: Dataset split to download (default: "train")
        num_proc: Number of processes for parallel processing
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate approximate number of examples based on size
    # Assuming average text size of ~2KB per example
    bytes_per_example = 2048
    total_examples = int(total_size_gb * 1e9 / bytes_per_example)
    tokenizer_examples = int(tokenizer_size_gb * 1e9 / bytes_per_example)
    examples_per_lang = total_examples // len(languages)
    tokenizer_examples_per_lang = tokenizer_examples // len(languages)

    print(f"Downloading FineWeb 2 for {len(languages)} languages")
    print(f"Target total size: {total_size_gb} GB ({total_examples:,} examples)")
    print(f"Target tokenizer size: {tokenizer_size_gb} GB ({tokenizer_examples:,} examples)")
    print(f"Examples per language: ~{examples_per_lang:,} (total), ~{tokenizer_examples_per_lang:,} (tokenizer)")

    all_data = []
    tokenizer_data = []

    for lang in tqdm(languages, desc="Processing languages"):
        try:
            print(f"\nDownloading {lang}...")

            # Load dataset for this language
            # Note: Adjust the dataset path based on actual FineWeb 2 structure
            dataset = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name=lang,
                split=split,
                streaming=True,
            )

            # Collect examples
            lang_data = []
            lang_tokenizer_data = []

            for idx, example in enumerate(dataset):
                if idx >= examples_per_lang:
                    break

                text = example.get("text", "")
                if text:
                    lang_data.append({"text": text, "language": lang})

                    # Add to tokenizer data if within limit
                    if idx < tokenizer_examples_per_lang:
                        lang_tokenizer_data.append({"text": text, "language": lang})

            all_data.extend(lang_data)
            tokenizer_data.extend(lang_tokenizer_data)

            print(f"  Collected {len(lang_data):,} examples for {lang}")

        except Exception as e:
            print(f"  Error downloading {lang}: {e}")
            continue

    # Save the data
    print(f"\nSaving data to {output_path}")

    # Save full training data
    train_file = output_path / "train_data.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for example in tqdm(all_data, desc="Writing training data"):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Save tokenizer training data
    tokenizer_file = output_path / "tokenizer_data.jsonl"
    with open(tokenizer_file, "w", encoding="utf-8") as f:
        for example in tqdm(tokenizer_data, desc="Writing tokenizer data"):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    # Save metadata
    metadata = {
        "languages": languages,
        "total_size_gb": total_size_gb,
        "tokenizer_size_gb": tokenizer_size_gb,
        "total_examples": len(all_data),
        "tokenizer_examples": len(tokenizer_data),
        "examples_per_language": {
            lang: sum(1 for ex in all_data if ex["language"] == lang)
            for lang in languages
        }
    }

    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Downloaded {len(all_data):,} total examples")
    print(f"✓ Created tokenizer subset with {len(tokenizer_data):,} examples")
    print(f"✓ Saved to {output_path}")
    print(f"  - Training data: {train_file}")
    print(f"  - Tokenizer data: {tokenizer_file}")
    print(f"  - Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FineWeb 2 data for tokenizer and LM training"
    )
    parser.add_argument(
        "--lang_set",
        type=str,
        default="LANG_SET_20",
        choices=list(LANG_SETS.keys()),
        help="Predefined language set to use",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="Custom list of language codes (overrides --lang_set)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--total_size_gb",
        type=float,
        required=True,
        help="Total size of data to download for LM training (in GB)",
    )
    parser.add_argument(
        "--tokenizer_size_gb",
        type=float,
        default=2.0,
        help="Size of data for tokenizer training (in GB)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to download",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing",
    )

    args = parser.parse_args()

    # Determine languages to use
    if args.languages:
        languages = args.languages
    else:
        languages = LANG_SETS[args.lang_set]

    print(f"Languages: {', '.join(languages)}")

    # Download data
    download_fineweb_subset(
        languages=languages,
        output_dir=args.output_dir,
        total_size_gb=args.total_size_gb,
        tokenizer_size_gb=args.tokenizer_size_gb,
        split=args.split,
        num_proc=args.num_proc,
    )


if __name__ == "__main__":
    main()
