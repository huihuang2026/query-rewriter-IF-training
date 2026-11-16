#!/usr/bin/env bash
# Serve the W4A16 model on one GPU. Edit/comment the settings directly.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=0
python_bin="$PWD/.venv/bin/python" # Local vLLM environment.
export PATH="$(dirname -- "$python_bin"):$PATH"
checkpoint="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
quantization_suffix="W4A16-GPTQ-calib10240"
model_name="${checkpoint}-${quantization_suffix}"
model="$PWD/cpkts/$model_name"
host="127.0.0.1"
port=8000
max_model_len=32768
gpu_memory_utilization=0.90
max_num_seqs=64
max_num_batched_tokens=8192

# Keep the exact launcher, package versions, GPU details, and selected kernel.
mkdir -p logs/quantization
server_log="logs/quantization/serve_w4a16_$(date +%Y%m%d_%H%M%S_%N).log"
cp -- "$PWD/quantization/serve_w4a16.sh" "${server_log%.log}_script.sh"
"$python_bin" -c 'from importlib.metadata import version; print({p: version(p) for p in ("vllm", "torch", "flashinfer-python")})' > "$server_log"
nvidia-smi --query-gpu=index,uuid,name,driver_version,power.limit --format=csv >> "$server_log"
printf 'Server log: %s\n' "$server_log"

# 1. Start the server in this terminal; wait until it reports ready.
# Quantization is detected from config.json; the KV cache remains BF16.
# Disable prefix caching so repeated/shared test prompts do not bias the comparison.
"$python_bin" -m vllm.entrypoints.cli.main serve "$model" \
    --served-model-name "$model_name" \
    --host "$host" --port "$port" --dtype bfloat16 --kv-cache-dtype auto \
    --tensor-parallel-size 1 --pipeline-parallel-size 1 --data-parallel-size 1 \
    --max-model-len "$max_model_len" --gpu-memory-utilization "$gpu_memory_utilization" \
    --max-num-seqs "$max_num_seqs" --max-num-batched-tokens "$max_num_batched_tokens" \
    --no-enable-prefix-caching --generation-config vllm --seed 42 --no-enable-log-requests \
    2>&1 | tee -a "$server_log"
