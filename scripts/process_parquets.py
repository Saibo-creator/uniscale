#!/usr/bin/env python3
"""
Inspect and convert parquet files to JSONL format.
Language is taken from each parquet file's stem name.

Usage:
    # Inspect all parquet files under data/tokenizer_training_dataset
    python scripts/process_parquets.py --input_dir data/tokenizer_training_dataset

    # Convert all data → produces out/train.jsonl and out/dev.jsonl (dev = 1%)
    python scripts/process_parquets.py \\
        --input_dir data/tokenizer_training_dataset \\
        --output_dir out/all

    # Convert a single subdataset
    python scripts/process_parquets.py \\
        --input_dir data/tokenizer_training_dataset/fineweb2 \\
        --output_dir out/fineweb2

Output JSONL format (train.jsonl and dev.jsonl written to output_dir):
    {"text": "Hello world", "language": "aai_Latn"}
    {"text": "Another sentence", "language": "aai_Latn"}
    {"text": "Bonjour le monde", "language": "fra_Latn"}
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

DEV_RATIO = 0.01  # 1% of rows go to dev


def find_parquets(input_dir: Path) -> list[Path]:
    """Find parquet files directly in input_dir, or recursively in subdirectories."""
    direct = sorted(input_dir.glob("*.parquet"))
    if direct:
        return direct
    return sorted(input_dir.glob("**/*.parquet"))


def get_text(row: dict) -> str | None:
    return row.get("text") or row.get("content") or None


def split_output_paths(output_dir: str) -> tuple[Path, Path]:
    d = Path(output_dir)
    return d / "train.jsonl", d / "dev.jsonl"


def inspect(input_dir: str) -> None:
    parquet_files = find_parquets(Path(input_dir))
    if not parquet_files:
        print(f"No .parquet files found under {input_dir}")
        return

    total_rows = 0
    print(f"Found {len(parquet_files)} parquet file(s) under {input_dir}\n")
    print(f"{'Language':<30} {'Rows':>10}  Columns")
    print("-" * 65)

    for pf in parquet_files:
        table = pq.read_table(pf)
        rows = table.num_rows
        cols = table.schema.names
        total_rows += rows
        print(f"{pf.stem:<30} {rows:>10}  {cols}")

    print("-" * 65)
    print(f"{'TOTAL':<30} {total_rows:>10}")


def convert(input_dir: str, output_dir: str) -> None:
    parquet_files = find_parquets(Path(input_dir))
    if not parquet_files:
        print(f"No .parquet files found under {input_dir}")
        return

    train_path, dev_path = split_output_paths(output_dir)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    train_total = dev_total = skipped = 0

    with open(train_path, "w", encoding="utf-8") as train_f, \
         open(dev_path, "w", encoding="utf-8") as dev_f:

        for pf in parquet_files:
            language = pf.stem
            table = pq.read_table(pf)
            rows_dict = table.to_pydict()
            n = table.num_rows

            lang_train = lang_dev = 0
            dev_every = max(1, round(1 / DEV_RATIO))  # every 100th row → dev

            for i in range(n):
                row = {col: rows_dict[col][i] for col in rows_dict}
                text = get_text(row)
                if not text or not text.strip():
                    skipped += 1
                    continue
                record = json.dumps({"text": text.strip(), "language": language}, ensure_ascii=False) + "\n"
                if i % dev_every == 0:
                    dev_f.write(record)
                    lang_dev += 1
                else:
                    train_f.write(record)
                    lang_train += 1

            train_total += lang_train
            dev_total += lang_dev
            print(f"  {language}: {lang_train} train / {lang_dev} dev")

    print(f"\nDone.")
    print(f"  train: {train_total} rows → {train_path}")
    print(f"  dev:   {dev_total} rows → {dev_path}")
    if skipped:
        print(f"  skipped {skipped} empty rows")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect or convert parquet files to JSONL (train/dev split)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .parquet files (or subdirs with .parquet files)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (e.g. out/all → out/all/train.jsonl + out/all/dev.jsonl). "
             "If omitted, only inspect.",
    )
    args = parser.parse_args()

    if args.output_dir:
        convert(args.input_dir, args.output_dir)
    else:
        inspect(args.input_dir)


if __name__ == "__main__":
    main()
