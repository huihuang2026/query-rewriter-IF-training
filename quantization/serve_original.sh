#!/usr/bin/env bash
# Serve the original train02 model in BF16. Comment/uncomment individual steps.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=2
merge_python="$PWD/.venv-quantization/bin/python" # PyTorch/PEFT environment for merging.
python_bin="$HOME/miniconda3/bin/python" # vLLM environment for serving.
checkpoint="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
base_model="$PWD/models/Qwen3-4B-Instruct-2507"
adapter="$PWD/ckpts/$checkpoint/final"
model_name="${checkpoint}-BF16"
model="$PWD/ckpts/$model_name"
host="127.0.0.1"
port=8002
max_model_len=32768
gpu_memory_utilization=0.90
max_num_seqs=64
max_num_batched_tokens=8192

# # 1. One-time merge of the original BF16 base and trained adapter, without quantization.
# # Uses the same merge procedure as FP8/W4A16. Comment out after the BF16 model is saved.
# "$merge_python" -u quantization/merge_original.py \
#     --base-model "$base_model" --adapter "$adapter" --output-dir "$model"

# 2. Serve the merged model, using the same engine settings as FP8/W4A16.
"$python_bin" -m vllm.entrypoints.cli.main serve "$model" \
    --served-model-name "$model_name" \
    --host "$host" --port "$port" --dtype bfloat16 --kv-cache-dtype auto \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --max-model-len "$max_model_len" --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_num_batched_tokens" \
    --no-enable-prefix-caching --generation-config vllm --seed 42 --no-enable-log-requests
