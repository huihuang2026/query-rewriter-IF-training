#!/usr/bin/env bash
# Comment/uncomment the steps you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# set -a
# source .envs.apikeys
# set +a

python_bin=python
input_file="data/query_rewriter_exported_cleaned.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
pairs_file="data/query_rewriter_seed_pairs.jsonl"
seed_file="data/query_rewriter_exported_customer_max10_processed.json"
teacher_model="gemini-3.8-flash"
max_per_customer=10
seed=42

# 1. Select authentic customer instructions and generate rewrites for both conditions.
# Add --prepare-only to select the input without making API calls.
"$python_bin" -u scripts/src/construct_seed_data.py generate \
    --input "$input_file" --test "$test_file" --output "$pairs_file" \
    --max-per-customer "$max_per_customer" --seed "$seed" \
    --model "$teacher_model" --thinking-level low --batch-size 100 --workers 20

# 2. Convert the pairs to the seed examples used by self-instruction.
"$python_bin" -u scripts/src/construct_seed_data.py process \
    --input "$pairs_file" --output "$seed_file"
