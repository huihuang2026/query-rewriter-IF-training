#!/usr/bin/env bash
# Calibrate and quantize the merged BF16 checkpoint to NVFP4 W4A4.
# Edit the settings below before running. Requires quantization/requirements.txt.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=0
python_bin="$PWD/.venv-quantization/bin/python"
export PATH="$(dirname -- "$python_bin"):$PATH"
checkpoint="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
model_path="$PWD/cpkts/${checkpoint}-BF16"
output_dir="$PWD/cpkts/${checkpoint}-NVFP4"
num_samples=10240 # Total prompts, balanced between both instruction conditions.
seed=42
max_seq_length=4096 # Calibration cap; does not change inference context length.
calibration_file="$PWD/data/quantization/calibration_${num_samples}_seed${seed}.jsonl"

# Optional: prepare calibration prompts from your training export first.
# "$python_bin" -u quantization/sample_calibration.py \
#     --input /path/to/query_rewriter_combined_selfins_unused_instructions.json \
#     --output "$calibration_file" --num-samples "$num_samples" --seed "$seed"

# Use saved training calibration prompts; stop other GPU workloads before running.
# Extra arguments are forwarded, e.g. bash quantization/quantize_nvfp4.sh --validate-only.
"$python_bin" -u quantization/quantize_nvfp4.py \
    --model-path "$model_path" --output-dir "$output_dir" \
    --calibration-file "$calibration_file" --num-samples "$num_samples" \
    --max-seq-length "$max_seq_length" --seed "$seed" "$@"
