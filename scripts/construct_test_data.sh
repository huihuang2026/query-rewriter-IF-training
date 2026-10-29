#!/usr/bin/env bash
# Construct the test set by reviewing language, quality, and retrieval need.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/construct_test_data.sh"
    echo "Uses input_file if present; otherwise cleans raw_exported and samples the review pool."
    echo "Edit input_file, raw_exported, pool_size, final_file, and limit directly in this script."
    exit 0
fi
python_bin="$PWD/.venv/bin/python"
input_file="data/query_rewriter_exported_sample10k.json"
final_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
output_dir="data"
raw_exported="data/export-data"
pool_size=10000
limit="1000"
seed=42 # Change this to select a different random subset.
# Keep review journals separate from the published held-out test file.
suffix="exported_sample10k_random1k"
random_file="$output_dir/query_rewriter_${suffix}.json"
language_filtered_file="$output_dir/query_rewriter_${suffix}_language_filtered.json"
quality_scored_file="$output_dir/query_rewriter_${suffix}_quality_scored.json"
quality_ranked_file="$output_dir/query_rewriter_${suffix}_quality_ranked.json"
quality_filtered_file="$output_dir/query_rewriter_${suffix}_quality_filtered.json"
retrieval_file="$output_dir/query_rewriter_${suffix}_retrieval_necessity.json"
batch_size=50
workers=10
quality_score=3 # Keep scores strictly above this cutoff.
model="gpt-6-luna"

export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}"

# Prepare the initial review pool here when it has not been created yet.
if [[ ! -s "$input_file" ]]; then
    cleaned_file="$output_dir/query_rewriter_exported_cleaned.json"
    mkdir -p "$output_dir" "$(dirname -- "$input_file")"
    if [[ ! -s "$cleaned_file" ]]; then
        "$python_bin" scripts/src/prepare_test_data.py \
            "$raw_exported" "$cleaned_file" --skip-invalid
    fi
    "$python_bin" scripts/src/prepare_test_data.py \
        "$cleaned_file" "$input_file" --sample --sample-size "$pool_size" --seed "$seed"
fi

# Randomly select 1k records from the 10k pool and annotate their language.
"$python_bin" scripts/src/construct_test_data.py language \
    --input "$input_file" --output "$random_file" \
    --limit "$limit" --seed "$seed" --overwrite

"$python_bin" scripts/src/construct_test_data.py filter-language \
    --input "$random_file" --output "$language_filtered_file"

"$python_bin" scripts/src/construct_test_data.py quality \
    --input "$language_filtered_file" --output "$quality_scored_file" \
    --limit "$limit" --overwrite \
    --model "$model" --batch-size "$batch_size" --workers "$workers"

"$python_bin" scripts/src/construct_test_data.py rank \
    --input "$quality_scored_file" --output "$quality_ranked_file"

"$python_bin" scripts/src/construct_test_data.py filter-quality \
    --input "$quality_ranked_file" --output "$quality_filtered_file" --score "$quality_score"

"$python_bin" scripts/src/construct_test_data.py retrieval \
    --input "$quality_filtered_file" --output "$retrieval_file" \
    --limit "$limit" --overwrite \
    --model "$model" --batch-size "$batch_size" --workers "$workers"

# Export the reviewed test set and print counts for the filtering stages.
"$python_bin" scripts/src/construct_test_data.py export \
    --input "$retrieval_file" --output "$final_file" \
    --random-file "$random_file" \
    --language-filtered-file "$language_filtered_file" \
    --quality-filtered-file "$quality_filtered_file"
