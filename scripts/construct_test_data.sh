#!/usr/bin/env bash
# Comment/uncomment the steps you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Uncomment to load API keys from the project root.
# set -a
# source .envs.apikeys
# set +a

python_bin=python
input_file="data/query_rewriter_exported_cleaned.json"
output_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
prefix="data/query_rewriter_test"
judge_model="gpt-6-luna"
pool_size=10000
review_size=1000
seed=42

# 1. Sample a customer-balanced candidate pool.
"$python_bin" -u scripts/src/prepare_test_data.py "$input_file" "${prefix}_pool.jsonl" \
    --sample --sample-size "$pool_size" --seed "$seed"

# 2. Randomly select review_size conversations and identify their languages.
"$python_bin" -u scripts/src/construct_test_data.py language \
    --input "${prefix}_pool.jsonl" --output "${prefix}_language.jsonl" \
    --limit "$review_size" --seed "$seed"

# 3. Filter languages.
"$python_bin" -u scripts/src/construct_test_data.py filter-language \
    --input "${prefix}_language.jsonl" --output "${prefix}_language_filtered.jsonl"

# 4. Judge conversation quality, then retain scores above 3.
"$python_bin" -u scripts/src/construct_test_data.py quality \
    --input "${prefix}_language_filtered.jsonl" --output "${prefix}_quality.jsonl" \
    --model "$judge_model" --limit "$review_size" --batch-size 50 --workers 10

"$python_bin" -u scripts/src/construct_test_data.py rank \
    --input "${prefix}_quality.jsonl" --output "${prefix}_ranked.jsonl"

"$python_bin" -u scripts/src/construct_test_data.py filter-quality \
    --input "${prefix}_ranked.jsonl" --output "${prefix}_quality_filtered.jsonl" --score 3

# 5. Judge whether the next response needs retrieval.
"$python_bin" -u scripts/src/construct_test_data.py retrieval \
    --input "${prefix}_quality_filtered.jsonl" --output "${prefix}_retrieval.jsonl" \
    --model "$judge_model" --limit "$review_size" --batch-size 50 --workers 10

# 6. Export the final test set.
"$python_bin" -u scripts/src/construct_test_data.py export \
    --input "${prefix}_retrieval.jsonl" --output "$output_file" \
    --random-file "${prefix}_language.jsonl" \
    --language-filtered-file "${prefix}_language_filtered.jsonl" \
    --quality-filtered-file "${prefix}_quality_filtered.jsonl"
