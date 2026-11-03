#!/usr/bin/env bash
# Generate and score both instruction conditions, then calculate metrics.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Edit these settings directly for each evaluation.
model_name="Qwen3-4B-Instruct-2507-700K-SFT-34K-DPO-app5-rej2-34K-DPO-length-control"
test_python="$HOME/miniconda3/bin/python"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
judge_model="gpt-6-astra"
test_limit="NULL"
gpu=0

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash test_query_rewriter.sh"
    echo "Model: $model_name (edit model_name in this script to change it)."
    echo "Scores quality without user instructions and following with them."
    echo "Generates both instruction conditions, scores them, and prints metrics."
    exit 0
fi

if [[ -f "$HOME/api-key.txt" ]]; then
    source "$HOME/api-key.txt"
fi

# Full model names resolve directly; adapters use the shared Qwen base model.
model="models/$model_name"
adapter_args=()
if [[ -f "ckpts/$model_name/final/adapter_config.json" ]]; then
    model="models/Qwen3-4B-Instruct-2507"
    adapter_args=(--lora-path "ckpts/$model_name/final")
elif [[ -f "ckpts/$model_name/final/config.json" ]]; then
    model="ckpts/$model_name/final"
elif [[ ! -f "$model/config.json" ]]; then
    echo "Model not found: $model_name" >&2
    exit 1
fi

test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}_both_instruction_conditions"
mkdir -p outputs
# Prevent two runs from writing to the same model's output files together.
exec 9>"${output_prefix}.lock"
if ! flock -n 9; then
    echo "An evaluation for $model_name is already running." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="$gpu" "$test_python" scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$model" --model-name "$model_name" "${adapter_args[@]}" \
    --limit "$test_limit" --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
