#!/usr/bin/env bash
# Serve the W4A16 model with Marlin on one GPU. Edit/comment the settings directly.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=3
python_bin="$HOME/miniconda3/bin/python" # vLLM environment.
checkpoint="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
quantization_suffix="W4A16-GPTQ-calib10240"
model_name="${checkpoint}-${quantization_suffix}"
model="$PWD/ckpts/$model_name"
host="127.0.0.1"
port=8003
max_model_len=32768
gpu_memory_utilization=0.90
max_num_seqs=64
max_num_batched_tokens=8192

# 1. Start this Marlin server on GPU 3 and port 8003; wait until it reports ready.
# Quantization is detected from config.json; Marlin handles the W4A16 linear layers.
# Use benchmark_w4a16_marlin.sh to query this server once it reports ready.
"$python_bin" -m vllm.entrypoints.cli.main serve "$model" \
    --served-model-name "$model_name" \
    --host "$host" --port "$port" --dtype bfloat16 --kv-cache-dtype auto \
    --linear-backend marlin \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --max-model-len "$max_model_len" --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_num_batched_tokens" \
    --no-enable-prefix-caching --generation-config vllm --seed 42 --no-enable-log-requests
