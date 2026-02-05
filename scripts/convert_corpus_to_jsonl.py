"""
Convert corpus_downloader output (per-language .txt files) to JSONL format.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm


def convert_corpus_to_jsonl(
    corpus_dir: str,
    output_file: str,
    split: str = "train"
):
    """
    Convert per-language corpus files to a single JSONL file.

    Args:
        corpus_dir: Directory containing language subdirectories (e.g., data/raw/tokenizer_corpus)
        output_file: Output JSONL file path
        split: Which split to convert (train, val, or test)
    """
    corpus_path = Path(corpus_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all language directories
    lang_dirs = [d for d in corpus_path.iterdir() if d.is_dir()]

    if not lang_dirs:
        print(f"No language directories found in {corpus_dir}")
        return

    print(f"Found {len(lang_dirs)} languages")

    total_lines = 0
    total_bytes = 0

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for lang_dir in tqdm(lang_dirs, desc="Converting languages"):
            lang = lang_dir.name
            txt_file = lang_dir / f"{split}.txt"

            if not txt_file.exists():
                print(f"  Warning: {txt_file} not found, skipping")
                continue

            # Read and convert to JSONL
            with open(txt_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    text = line.rstrip('\n')  # Remove trailing newline
                    if text:  # Skip empty lines
                        json_obj = {
                            "text": text,
                            "language": lang
                        }
                        out_f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                        total_lines += 1
                        total_bytes += len(text.encode('utf-8'))

    print(f"\n✓ Converted {total_lines:,} lines ({total_bytes/1e9:.2f} GB)")
    print(f"✓ Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert corpus_downloader output to JSONL format"
    )
    parser.add_argument(
        "--tokenizer_corpus_dir",
        type=str,
        default="data/raw/tokenizer_corpus",
        help="Directory containing tokenizer corpus",
    )
    parser.add_argument(
        "--main_corpus_dir",
        type=str,
        default="data/raw/main_corpus",
        help="Directory containing main corpus",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Output directory for JSONL files",
    )

    args = parser.parse_args()

    print("Converting tokenizer corpus...")
    convert_corpus_to_jsonl(
        args.tokenizer_corpus_dir,
        f"{args.output_dir}/tokenizer_data.jsonl",
        split="train"
    )

    print("\nConverting main corpus (train)...")
    convert_corpus_to_jsonl(
        args.main_corpus_dir,
        f"{args.output_dir}/train_data.jsonl",
        split="train"
    )

    print("\nConverting main corpus (val)...")
    convert_corpus_to_jsonl(
        args.main_corpus_dir,
        f"{args.output_dir}/val_data.jsonl",
        split="val"
    )

    print("\nConverting main corpus (test)...")
    convert_corpus_to_jsonl(
        args.main_corpus_dir,
        f"{args.output_dir}/test_data.jsonl",
        split="test"
    )

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)


if __name__ == "__main__":
    main()
