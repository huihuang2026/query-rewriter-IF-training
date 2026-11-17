#!/usr/bin/env bash
# Quantize and evaluate FP8. Comment/uncomment individual steps as needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=1
python_bin="$PWD/.venv-quantization/bin/python"
test_python="$PWD/.venv/bin/python" # Local vLLM/OpenAI environment, separate from llmcompressor.
export PATH="$(dirname -- "$test_python"):$PATH"
base_model="models/Qwen3-4B-Instruct-2507"
model_name="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
adapter="ckpts/$model_name/final"
quantized_name="${model_name}-FP8-DYNAMIC"
output_dir="ckpts/$quantized_name"
seed=42
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
judge_model="gpt-6-astra"
limit="NULL"
max_new_tokens=256
output_prefix="outputs/quantization/query_rewriter_${quantized_name}_both_instruction_conditions"

# 1. Merge the BF16 base and train02 adapter, then apply FP8_DYNAMIC quantization.
# FP8 weights use per-channel scales; activation scales are computed per token at inference.
# No calibration samples or sequence-length limit are needed for this scheme.
# Comment out this command once the FP8 checkpoint has been saved.
"$python_bin" -u quantization/quantize_fp8.py \
    --base-model "$base_model" --adapter "$adapter" --output-dir "$output_dir" --seed "$seed"

# 2. Generate answers with and without user-defined instructions using the saved FP8 model.
mkdir -p outputs/quantization
"$test_python" -u scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$output_dir" --model-name "$quantized_name" \
    --max-new-tokens "$max_new_tokens" --limit "$limit" --both-instruction-conditions

# 3. Load API keys and judge the generated answers under matching instruction conditions.
set -a
source "$PWD/.envs.apikeys"
set +a

"$test_python" -u scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

# 4. Calculate and save metrics.
"$test_python" -u scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
