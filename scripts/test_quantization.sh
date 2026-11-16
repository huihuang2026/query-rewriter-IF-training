#!/usr/bin/env bash
# Evaluate merged GPTQ weights. Comment/uncomment each step as needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

set -a
source "$PWD/.envs.apikeys"
set +a

export CUDA_VISIBLE_DEVICES=0
python_bin=python # Use the vLLM environment, separate from llmcompressor.
model_name="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch-W4A16-GPTQ-calib10240"
model="ckpts/$model_name"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
judge_model="gpt-6-astra"
limit="NULL"
max_new_tokens=256
output_prefix="outputs/quantization/query_rewriter_${model_name}_both_instruction_conditions"
mkdir -p outputs/quantization

# 1. Generation already completed. Uncomment this block to generate again.
# "$python_bin" -u scripts/src/test_query_rewriter.py generate \
#     --input "$test_file" --output "${output_prefix}_generated.json" \
#     --model-path "$model" --model-name "$model_name" \
#     --max-new-tokens "$max_new_tokens" --limit "$limit" --both-instruction-conditions

# 2. Judge the generated answers.
"$python_bin" -u scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

# 3. Calculate metrics.
"$python_bin" -u scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
