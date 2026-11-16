#!/usr/bin/env bash
# Comment/uncomment sampling and quantization independently.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=0
python_bin="$PWD/.venv-quantization/bin/python"
base_model="models/Qwen3-4B-Instruct-2507"
model_name="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
adapter="ckpts/$model_name/final"
input_file="data/trash-bin/final-mixed-training/query_rewriter_combined_selfins_unused_instructions.json"
num_samples=10240 # Total prompts: 5120 with instructions and 5120 without.
output_dir="ckpts/${model_name}-W4A16-GPTQ-calib${num_samples}"
seed=42
max_seq_length=4096 # Calibration cap; does not set the inference context limit.
calibration_file="data/quantization/calibration_${num_samples}_seed${seed}.jsonl"

# 1. Calibration prompts already generated. Uncomment to sample them again.
# "$python_bin" -u quantization/sample_calibration.py \
#     --input "$input_file" --output "$calibration_file" \
#     --num-samples "$num_samples" --seed "$seed"

# 2. Merge the adapter and quantize using the saved calibration prompts.
"$python_bin" -u quantization/quantize_w4a16.py \
    --calibration-file "$calibration_file" --base-model "$base_model" \
    --adapter "$adapter" --output-dir "$output_dir" \
    --num-samples "$num_samples" --max-seq-length "$max_seq_length" --seed "$seed"
