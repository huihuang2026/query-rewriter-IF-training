#!/usr/bin/env bash
# Rebuild customer-capped training data from authentic export instructions.
# Run from any directory; comment individual steps below to run them separately.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/construct_seed_data.sh [generate options]"
    echo "Cleans exported data if needed, selects customer-capped records, generates paired rewrites, and prepares SFT data."
    echo "Default: max 10 records per customer, selected directly from cleaned exported data."
    echo "Edit the paths/settings below, or comment individual command blocks."
    exit 0
fi

export GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}"

python_bin="$PWD/.venv/bin/python"
data_dir="data"
raw_exported="$data_dir/export-data"
input_file="$data_dir/query_rewriter_exported_cleaned.json"
test_file="$data_dir/query_rewriter_exported_sample10k_random1k_filtered.json"
max_per_customer=10
seed=42
model="gemini-3.8-flash"
batch_size=100
workers=20
# Optional bulk Gemini pairs from an earlier run; missing files are ignored.
# Set this to an existing paired output to reuse its matching completed responses.
reuse_file="$data_dir/query_rewriter_exported_train_gemini_pairs.json"
output_file="$data_dir/query_rewriter_exported_train_customer_max${max_per_customer}.json"
processed_file="$data_dir/query_rewriter_exported_customer_max${max_per_customer}_processed.json"

# Step 1: Clean original exported data, sharing the same cleaned source as test construction.
if [[ ! -s "$input_file" ]]; then
    "$python_bin" -u scripts/src/prepare_test_data.py "$raw_exported" "$input_file"
fi

# Step 2: Exclude held-out instructions/conversations, then select up to ten records
# per customer directly from cleaned exported data, preferring completed reusable pairs.
# Resume uses the fixed selection in ${output_file%.json}_input.json.
# Each pair receives independent Gemini answers with and without its authentic instruction.
"$python_bin" -u scripts/src/construct_seed_data.py generate \
    --input "$input_file" --test "$test_file" --reuse "$reuse_file" --output "$output_file" \
    --max-per-customer "$max_per_customer" --seed "$seed" \
    --model "$model" --thinking-level low \
    --batch-size "$batch_size" --workers "$workers" "$@"

# Step 3: Convert the two real response branches to SFT examples; drop failed/empty responses.
# Ten conversation records can yield up to twenty SFT examples per customer.
"$python_bin" -u scripts/src/construct_seed_data.py process \
    --input "$output_file" --output "$processed_file"
