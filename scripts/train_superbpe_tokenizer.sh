#!/bin/bash
# Simple script to train a superBPE tokenizer

vocab_size=128000
num_bytes=$((10**9))  # 1GB
regex_string="[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
corpus_dir="${PROJECT_ROOT}/data/raw/tokenizer_corpus/en"
output_dir="${PROJECT_ROOT}/out/tokenizers/superbpe_128k_1G"

cd "${PROJECT_ROOT}/tmp/superbpe"

python -m train_tokenizer \
    --output_dir "$output_dir" \
    --corpus_dir "$corpus_dir" \
    --num_bytes $num_bytes \
    --vocab_size $vocab_size \
    --regex_string "$regex_string"
