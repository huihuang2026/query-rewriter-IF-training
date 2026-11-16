#!/usr/bin/env bash
# Measure speed at 16, 64, and 256 QPS. Comment/uncomment the calls below as needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python_bin="$HOME/miniconda3/bin/python" # vLLM environment.
# One-time benchmark dependencies (already installed on this machine):
# "$python_bin" -m pip install pandas matplotlib seaborn datasets scipy plotly
checkpoint="Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
model_name="${checkpoint}-FP8-DYNAMIC"
host="127.0.0.1"
port=8001
model_label="FP8"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
tokenizer="models/Qwen3-4B-Instruct-2507"
result_dir="outputs/quantization/speed/fp8"
benchmark_file="${result_dir}/test_prompts.jsonl"
num_prompts=1032 # All 516 test conversations, each with and without instructions.
max_new_tokens=256
warmup_requests=10
seed=42
run_timestamp="$(date +%Y%m%d_%H%M%S)"
preparation_log="${result_dir}/prepare_${run_timestamp}.log"
mkdir -p "$result_dir"

# 1. Prepare both instruction conditions once for all three rates.
"$python_bin" -u quantization/prepare_speed_data.py \
    --input "$test_file" --output "$benchmark_file" \
    --tokenizer "$tokenizer" --max-new-tokens "$max_new_tokens" > "$preparation_log"

# 2. Benchmark one rate, then print/save its four latency metrics.
# No client concurrency cap. Each run finishes before the next rate starts.
# Warmups are excluded; responses stop normally at EOS.
run_benchmark() {
    local request_rate="$1"
    local run_name="qps${request_rate}_${run_timestamp}"
    local result_file="${run_name}_raw.json"
    local log_file="${result_dir}/${run_name}.log"

    "$python_bin" -m vllm.entrypoints.cli.main bench serve \
        --host "$host" --port "$port" --model "$model_name" --backend openai --endpoint /v1/completions \
        --tokenizer "$tokenizer" --dataset-name custom --dataset-path "$benchmark_file" \
        --skip-chat-template --disable-shuffle --no-oversample --num-prompts "$num_prompts" \
        --custom-output-len "$max_new_tokens" --temperature 0 --seed "$seed" \
        --extra-body '{"add_special_tokens": false}' \
        --request-rate "$request_rate" --num-warmups "$warmup_requests" \
        --percentile-metrics ttft,tpot --metric-percentiles 95 --disable-tqdm \
        --save-result --result-dir "$result_dir" --result-filename "$result_file" --label "qps${request_rate}" \
        > "$log_file"

    "$python_bin" -u quantization/report_speed.py \
        --input "${result_dir}/${result_file}" --label "$model_label" \
        | tee "${result_dir}/${run_name}_summary.md"
}

# 3. Run the three rates in order. Comment out any rate you do not need.
run_benchmark 16
run_benchmark 64
run_benchmark 256
