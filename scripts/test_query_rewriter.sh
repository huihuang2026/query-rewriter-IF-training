#!/usr/bin/env bash
# Comment/uncomment generation, judging, and metrics independently.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

set -a
source "$PWD/.envs.apikeys"
set +a

export CUDA_VISIBLE_DEVICES=0
python_bin=python
model_name="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
base_model="models/Qwen3-4B-Instruct-2507"
adapter="ckpts/$model_name/final"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
judge_model="gpt-6-astra"
limit="NULL"
max_new_tokens=256
# Separate outputs for this repeat evaluation of the original train02 adapter.
output_dir="outputs/rerun"
output_prefix="${output_dir}/query_rewriter_${model_name}_both_instruction_conditions"
mkdir -p "$output_dir"

# 1. Generate answers with and without user-defined instructions.
"$python_bin" -u scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$base_model" --lora-path "$adapter" --model-name "$model_name" \
    --max-new-tokens "$max_new_tokens" --limit "$limit" --both-instruction-conditions

# 2. Judge the generated answers under their matching instruction conditions.
"$python_bin" -u scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

# 3. Calculate metrics.
"$python_bin" -u scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
