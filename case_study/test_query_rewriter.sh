#!/usr/bin/env bash
# Compare the DPO checkpoint and query-rewriter model on the same samples.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "$script_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"

dpo_model_name="Qwen3-4B-Instruct-2507-700K-SFT-34K-DPO-app5-rej2-17K-DPO-length-control"
dpo_base_model="$workspace_dir/models/Qwen3-4B-Instruct-2507"
dpo_adapter="$workspace_dir/ckpts/$dpo_model_name/final"
query_model_name="Query-Rewrite-Qwen3-4B-2507"
query_model="$workspace_dir/models/$query_model_name"

export CUDA_VISIBLE_DEVICES="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Explicit options retain single-model mode; no arguments run both models below.
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "With no arguments, runs these models sequentially on GPU ${CUDA_VISIBLE_DEVICES}:"
    echo "  $workspace_dir/ckpts/$dpo_model_name (final LoRA adapter)"
    echo "  $query_model"
    echo "Both write model columns to case_study/output/retrieval_issue_testset_no_analysis.csv; reruns overwrite them."
    echo "Explicit options below run a single model instead."
fi
if (( $# )); then
    exec "$python_bin" -B "$script_dir/test_query_rewriter.py" "$@"
fi

for required_file in "$dpo_base_model/config.json" "$dpo_adapter/adapter_config.json" \
    "$dpo_adapter/adapter_model.safetensors" "$query_model/config.json"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required model file not found: $required_file" >&2
        exit 1
    fi
done

echo "[1/2] Evaluating $dpo_model_name"
"$python_bin" -B "$script_dir/test_query_rewriter.py" \
    --model-path "$dpo_base_model" --model-name "$dpo_model_name" --lora-path "$dpo_adapter"

echo "[2/2] Evaluating $query_model_name"
"$python_bin" -B "$script_dir/test_query_rewriter.py" \
    --model-path "$query_model" --model-name "$query_model_name"
