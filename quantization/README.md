# Quantization and speed comparison

All scripts use the existing train02 checkpoint: 700K SFT + previous DPO/length control + IF4/CP4. Edit variables directly and comment/uncomment command blocks.

## Quantization

- `quantize_w4a16.sh`: merge the BF16 base and train02 LoRA, then run GPTQ W4A16 with the saved 10,240 calibration prompts. Output suffix: `-W4A16-GPTQ-calib10240`.
- `quantize_fp8.sh`: merge the same base/adapter and apply data-free `FP8_DYNAMIC`, then optionally generate, judge, and report quality metrics. Output suffix: `-FP8-DYNAMIC`.

Quantization uses `.venv-quantization/bin/python`. Serving and benchmarking use `$HOME/miniconda3/bin/python`, which has vLLM. Edit these paths for another machine.

The benchmark client also requires vLLM's `bench` extras: `pandas`, `matplotlib`, `seaborn`, `datasets`, `scipy`, and `plotly`. These are installed in the base vLLM environment on this machine. On another machine, install them with the same Python interpreter used by the benchmark scripts; commented installation commands are included in both clients.

## Speed only

There is a server/client pair for each format:

| Format | Server | Benchmark client | Port |
|---|---|---|---|
| Original BF16 | `serve_original.sh` | `benchmark_original.sh` | 8002 |
| FP8 | `serve_fp8.sh` | `benchmark_fp8.sh` | 8001 |
| W4A16 | `serve_w4a16.sh` | `benchmark_w4a16.sh` | 8000 |
| W4A16 Marlin | `serve_w4a16_marlin.sh` | `benchmark_w4a16_marlin.sh` | 8003 |

`serve_original.sh` first merges the original train02 adapter into a full BF16 model at `ckpts/<train02-name>-BF16`, without quantization. The helper `merge_original.py` uses the same BF16 base/adapter merge procedure as the two quantizers. This makes the baseline a full model and avoids runtime LoRA overhead affecting the comparison. Comment out step 1 once the merged model has been saved. Current server GPU settings are W4A16=0, FP8=1, and original=2; edit `CUDA_VISIBLE_DEVICES` for your machine.

They do not call an external judge or recalculate quality scores. The benchmark renders the same system/user prompts as the existing evaluation, with thinking disabled, and sends them to the local `/v1/completions` endpoint. `prepare_speed_data.py` handles that conversion; the timing and throughput measurements use [vLLM's benchmark client](https://docs.vllm.ai/en/v0.31.0/benchmarking/cli/).

1. In terminal 1, start FP8 on its configured GPU and wait for the server to report ready:

   ```bash
   cd /home/huanghui/WorkSpace
   bash quantization/serve_fp8.sh
   ```

2. In terminal 2, send the test set and measure speed:

   ```bash
   cd /home/huanghui/WorkSpace
   bash quantization/benchmark_fp8.sh
   ```

3. To measure W4A16, start `bash quantization/serve_w4a16.sh` in terminal 1, wait for readiness, then run `bash quantization/benchmark_w4a16.sh` in terminal 2. For the unquantized baseline, use `bash quantization/serve_original.sh` and `bash quantization/benchmark_original.sh` instead. Each client explicitly names its matching model. For a controlled comparison on one physical GPU, stop the previous server and set the same GPU index in each server script.

Each server exposes one GPU and uses tensor/pipeline/data parallel sizes of 1. All three models use the same BF16 KV cache, context limit, batching settings, and GPU-memory utilization. Quantization is detected from each checkpoint's configuration; the original BF16 model has none. Prefix caching is disabled so shared prompts and repeated runs do not reuse a previous run's cached prefixes. Keep these settings and the physical GPU unchanged for a controlled comparison.

All four clients (Original, FP8, W4A16, and W4A16 Marlin) prepare the prompts once, then benchmark **16, 64, and 256 QPS sequentially**. Each rate uses the same 1,032 requests: 516 conversations with both instruction conditions. That is 3,096 measured requests per script. Comment/uncomment the `run_benchmark 16`, `run_benchmark 64`, and `run_benchmark 256` calls to select rates. There is no client `--max-concurrency` cap. Requests can queue inside the server under load, and achieved throughput can remain below the target. Server scheduling limits are configured separately in the serving scripts.

Each run uses greedy decoding, a maximum of 256 generated tokens, and 10 warmup requests excluded from the measured run. Server startup and prompt preparation are also excluded. HTTP timing includes streaming, networking, and server work; chat-template rendering happens beforehand.

The terminal report and saved Markdown summary contain only these four metrics, aggregated over both instruction conditions:

| TTFT Mean (ms) | TTFT P95 (ms) | Per Token Mean (ms/token) | Per Token P95 (ms/token) |
|---|---|---|---|
| … | … | … | … |

“Per Token” is vLLM's TPOT: for each response with more than one output token, `(end-to-end latency - TTFT) / (output tokens - 1)`. The mean and P95 summarize those per-response values, excluding the first token. `report_speed.py` reads the completed vLLM result and prints the table; incomplete/failed runs produce an error instead of a summary.

Results stay under `outputs/quantization/speed/original/`, `outputs/quantization/speed/fp8/`, `outputs/quantization/speed/w4a16/`, and `outputs/quantization/speed/w4a16_marlin/`. Each rate saves `qps<rate>_<timestamp>_summary.md`, `qps<rate>_<timestamp>_raw.json`, and `qps<rate>_<timestamp>.log`; all three rates share a timestamp. Prompt preparation messages go to `prepare_<timestamp>.log`. Full benchmark diagnostics, including achieved throughput and request counts, are retained in the raw JSON/log instead of being printed as extra report metrics. Errors on stderr remain visible.

Responses stop normally at EOS, so this measures the actual query-rewriting workload. The models may produce different output lengths. For a fixed amount of decode work, add `--ignore-eos` to each client's benchmark command, forcing exactly 256 output tokens per request. Keep that setting identical across all three models and use a separate result directory for those runs.

The prepared prompts and a checksum/count manifest are written under `outputs/quantization/speed/`. They contain test conversations and are covered by the repository's `outputs/` ignore rule. No API keys are needed for this local benchmark.

## Comparing on L40S

The current workspace GPUs are H100 80GB HBM3. L40S uses the Ada architecture, has 48GB GDDR6 and 864GB/s memory bandwidth, and supports native FP8 Tensor Core operations ([NVIDIA specifications](https://www.nvidia.com/en-gb/data-center/l40s/)). vLLM supports FP8 W8A8 on Ada and Hopper, and W4A16 through supported quantization kernels ([FP8 support](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/), [quantization compatibility](https://docs.vllm.ai/en/latest/features/quantization/)).

Absolute speed and the relative FP8/W4A16 ranking can change with memory bandwidth, kernel selection, prompt length, and concurrency. W4A16's smaller weights may help memory-bound decoding; FP8 W8A8 may benefit compute-heavy prefill and larger batches. These are possible advantages, not measured results for this model. Benchmark both formats on the actual L40S using the same vLLM version and settings; H100 timings do not establish which format wins on L40S. The existing checkpoints can be reused on supported L40S software without requantizing solely for the GPU change.
