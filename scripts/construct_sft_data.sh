#!/usr/bin/env bash
# Two sources -> self-instructions -> teacher rewrites -> rejudging -> SFT data.
# Comment/uncomment the steps you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# set -a
# source .envs.apikeys
# set +a

python_bin=python
prepare_python="$PWD/.venv-finetune/bin/python"
base_model="models/Qwen3-4B-Instruct-2507"
previous_input="data/query_rewriter_gemini_annotated_cleaned.jsonl"
exported_input="data/query_rewriter_exported_sample400k.json"
seed_file="data/query_rewriter_exported_customer_max10_processed.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
previous_prefix="data/query_rewriter_previous_selfins"
exported_prefix="data/query_rewriter_exported_selfins"
processed_file="data/query_rewriter_sft_processed.jsonl"
combined_file="data/query_rewriter_combined_selfins.jsonl"
unused_file="data/query_rewriter_combined_selfins_unused_instructions.json"
instruction_model="gemini-3.5-flash-lite"
response_model="gemini-3.8-flash"
scoring_model="gpt-6-luna"
conversation_limit=100000
variants_per_sample=2
sft_pairs_per_source=50000
min_quality=4
min_following=3
seed=42

# 1. Deduplicate and sample exported conversations, excluding seed/test overlap.
"$python_bin" -u scripts/src/construct_sft_data.py dedup \
    --input data/query_rewriter_exported_cleaned.json --output "$exported_input" \
    --seeds "$seed_file" --test "$test_file" --sample-size 400000 --seed "$seed"

# 2. Generate self-instructions for each source, without generating rewrites yet.
"$python_bin" -u scripts/src/construct_sft_data.py construct-inst \
    --input "$previous_input" --output "${previous_prefix}.jsonl" \
    --seeds "$seed_file" --test "$test_file" --model "$instruction_model" \
    --variants-per-sample "$variants_per_sample" --limit "$conversation_limit" \
    --seed "$seed" --batch-size 100 --workers 20

"$python_bin" -u scripts/src/construct_sft_data.py construct-inst \
    --input "$exported_input" --output "${exported_prefix}.jsonl" \
    --seeds "$seed_file" --test "$test_file" --model "$instruction_model" \
    --variants-per-sample "$variants_per_sample" --limit "$conversation_limit" \
    --seed "$seed" --batch-size 100 --workers 20

# 3. Generate teacher rewrites with and without each user instruction.
"$python_bin" -u scripts/src/construct_sft_data.py construct-resp \
    --input "${previous_prefix}.jsonl" --output "${previous_prefix}.jsonl" \
    --seeds "$seed_file" --test "$test_file" --model "$response_model" \
    --limit "$conversation_limit" --seed "$seed" --batch-size 100 --workers 20

"$python_bin" -u scripts/src/construct_sft_data.py construct-resp \
    --input "${exported_prefix}.jsonl" --output "${exported_prefix}.jsonl" \
    --seeds "$seed_file" --test "$test_file" --model "$response_model" \
    --limit "$conversation_limit" --seed "$seed" --batch-size 100 --workers 20

# 4. Judge/rejudge quality and instruction following. Matching saved scores are reused.
# Changing scoring_model causes rejudging; run step 5 again afterward.
"$python_bin" -u scripts/src/construct_sft_data.py score \
    --input "${previous_prefix}.jsonl" --scored-output "${previous_prefix}_scored.jsonl" \
    --scoring-model "$scoring_model" --limit NULL --seed "$seed" --batch-size 100 --workers 20

"$python_bin" -u scripts/src/construct_sft_data.py score \
    --input "${exported_prefix}.jsonl" --scored-output "${exported_prefix}_scored.jsonl" \
    --scoring-model "$scoring_model" --limit NULL --seed "$seed" --batch-size 100 --workers 20

# 5. Filter and sample conversation pairs from each source.
"$prepare_python" -u scripts/src/prepare_sft_data.py \
    --input "${previous_prefix}_scored.jsonl" --output "${previous_prefix}_sft.jsonl" \
    --test "$test_file" --sample-pairs --instruction-count 1 --examples "$sft_pairs_per_source" \
    --min-query-quality "$min_quality" --min-instruction-following "$min_following" --seed "$seed"

"$prepare_python" -u scripts/src/prepare_sft_data.py \
    --input "${exported_prefix}_scored.jsonl" --output "${exported_prefix}_sft.jsonl" \
    --test "$test_file" --sample-pairs --instruction-count 1 --examples "$sft_pairs_per_source" \
    --min-query-quality "$min_quality" --min-instruction-following "$min_following" --seed "$seed"

# 6. Combine selected pairs, expand both conditions, and tokenize at 4096 tokens.
"$prepare_python" -u scripts/src/prepare_sft_data.py \
    --input "${previous_prefix}_sft.jsonl" "${exported_prefix}_sft.jsonl" \
    --output "$processed_file" --test "$test_file" --instruction-count 1 --examples NULL \
    --model "$base_model" --max-seq-length 4096 --seed "$seed" \
    --tokenized-output data/query_rewriter_sft_tokenized

# 7. Combine full journals, preserving unused variants for DPO and calibration.
"$python_bin" -u scripts/src/merge_jsonl.py \
    --input "${previous_prefix}.jsonl" "${exported_prefix}.jsonl" \
    --output "$combined_file" --source-group-key source_record_id

# 8. Extract instructions that were not selected for SFT.
"$prepare_python" -u scripts/src/construct_dpo_data.py extract \
    --input "$combined_file" --sft-data "$processed_file" --tokenizer "$base_model" \
    --test "$test_file" --output "$unused_file" --allow-missing-responses --seed "$seed"
