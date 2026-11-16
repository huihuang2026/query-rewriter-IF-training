# Quantization and speed comparison

All scripts use the existing train02 checkpoint: 700K SFT + previous DPO/length control + IF4/CP4. Edit variables directly and comment/uncomment command blocks.

## Quantization

- `quantize_w4a16.sh`: merge the BF16 base and train02 LoRA, then run GPTQ W4A16 with the saved 10,240 calibration prompts. Output suffix: `-W4A16-GPTQ-calib10240`.
- `quantize_fp8.sh`: merge the same base/adapter and apply data-free `FP8_DYNAMIC`, then optionally generate, judge, and report quality metrics. Output suffix: `-FP8-DYNAMIC`.

Quantization uses `.venv-quantization/bin/python`. All `serve_*.sh` and
`benchmark_*.sh` scripts use the project-local `.venv/bin/python` and add its
command directory to `PATH` for runtime compilation tools such as `ninja`.
Create it with Python 3.12 and install the serving dependencies:

```bash
sudo apt-get install -y python3.12-dev build-essential
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r quantization/requirements-vllm.txt
```

All server/client pairs use checkpoints and their bundled tokenizers under `cpkts/`.
All servers use GPU 0 on this machine. Run one server at a time: stop the previous
server before starting another, since each reserves 90% of GPU memory.

The `vllm[bench]` requirement also installs the benchmark dependencies, including `pandas`, `matplotlib`, `seaborn`, `datasets`, `scipy`, and `plotly`.

## NVFP4

Edit the paths and sample count in `quantize_nvfp4.sh`, then run
`bash quantization/quantize_nvfp4.sh`. It calls the Python quantizer described below.

`quantize_nvfp4.py` loads an already merged BF16 checkpoint (by default the
train02 `-BF16` model in `cpkts/`) and applies LLM Compressor's standard
`QuantizationModifier(scheme="NVFP4")` recipe. This is **W4A4 floating point**,
with groups of 16; it differs from the GPTQ INT4 W4A16 recipe. `lm_head` stays
BF16 and KV-cache quantization is not enabled. The result is saved in
compressed-tensors format alongside the tokenizer and a quantization manifest.
The default output is `cpkts/<train02-name>-NVFP4`; existing outputs are refused.

This recipe requires **calibration data** to estimate activation global scales.
Use representative training conversations, including both user-instruction
conditions, rather than the held-out speed/quality evaluation set. Assistant
answers or judge scores are not required. Reuse a saved calibration JSONL from
`sample_calibration.py`, or prepare one from an available training export:

```bash
python3 quantization/sample_calibration.py \
    --input /path/to/query_rewriter_combined_selfins_unused_instructions.json \
    --num-samples 1024 --seed 42
```

The shell launcher currently uses the saved
`data/quantization/calibration_10240_seed42.jsonl` (10,240 prompts).
For another machine, copy the calibration file and its `.manifest.json`, if
available, or supply a training export. `--num-samples` on the quantizer checks
the expected row count; it does not sample or discard rows.

Install dependencies in the separate quantization environment, leaving the
serving environment intact:

```bash
uv venv --python 3.12 .venv-quantization
uv pip install --python .venv-quantization/bin/python -r quantization/requirements.txt

# Check the inputs without loading a model or requiring quantization packages.
python3 quantization/quantize_nvfp4.py \
    --calibration-file data/quantization/calibration_1024_seed42.jsonl \
    --num-samples 1024 --validate-only

# Stop the serving process first to free GPU memory, then calibrate and save.
CUDA_VISIBLE_DEVICES=0 .venv-quantization/bin/python quantization/quantize_nvfp4.py \
    --calibration-file data/quantization/calibration_1024_seed42.jsonl \
    --num-samples 1024 --max-seq-length 4096
```

Use `--model-path` to select another local, merged, unquantized checkpoint and
`--output-dir` to override the destination. Prompt rendering matches evaluation:
thinking disabled, generation prefix enabled, and no duplicate special tokens.
Long prompts are truncated at the calibration token cap and counted in the
manifest. The cap does not change the inference context length.

Recipe reference: [LLM Compressor NVFP4 example](https://github.com/vllm-project/llm-compressor/blob/0.15.0/examples/quantization_w4a4_fp4/llama3_example.py).

## Speed only

There is a server/client pair for each format:

| Format | Server | Benchmark client | Port |
|---|---|---|---|
| Original BF16 | `serve_original.sh` | `benchmark_original.sh` | 8002 |
| FP8 | `serve_fp8.sh` | `benchmark_fp8.sh` | 8001 |
| W4A16 | `serve_w4a16.sh` | `benchmark_w4a16.sh` | 8000 |
| W4A16 Marlin | `serve_w4a16_marlin.sh` | `benchmark_w4a16_marlin.sh` | 8003 |
| NVFP4 Auto | `serve_nvfp4.sh` | `benchmark_nvfp4.sh` | 8004 |

After NVFP4 quantization completes, run `bash quantization/serve_nvfp4.sh` and
wait for readiness, then run `bash quantization/benchmark_nvfp4.sh` in another
terminal. The server selects `--linear-backend auto` and loads
`cpkts/<train02-name>-NVFP4`. Results are saved under
`outputs/quantization/speed/nvfp4_auto/`.

`serve_original.sh` serves the existing merged BF16 model at `cpkts/<train02-name>-BF16`. Its optional merge command is commented out; enable it only when the original base model, adapter, and quantization environment are available. The helper `merge_original.py` uses the same BF16 base/adapter merge procedure as the two quantizers. This makes the baseline a full model and avoids runtime LoRA overhead affecting the comparison. All server scripts set `CUDA_VISIBLE_DEVICES=0`; edit it for another machine.

They do not call an external judge or recalculate quality scores. The benchmark renders the same system/user prompts as the existing evaluation, with thinking disabled, and sends them to the local `/v1/completions` endpoint. `prepare_speed_data.py` handles that conversion; the timing and throughput measurements use [vLLM's benchmark client](https://docs.vllm.ai/en/v0.31.0/benchmarking/cli/).

1. In terminal 1, start FP8 on its configured GPU and wait for the server to report ready:

   ```bash
   cd /home/huanghui/query-rewriter-IF-training
   bash quantization/serve_fp8.sh
   ```

2. In terminal 2, send the test set and measure speed:

   ```bash
   cd /home/huanghui/query-rewriter-IF-training
   bash quantization/benchmark_fp8.sh
   ```

3. To measure W4A16, start `bash quantization/serve_w4a16.sh` in terminal 1, wait for readiness, then run `bash quantization/benchmark_w4a16.sh` in terminal 2. For the unquantized baseline, use `bash quantization/serve_original.sh` and `bash quantization/benchmark_original.sh` instead. Each client explicitly names its matching model. For a controlled comparison on one physical GPU, stop the previous server and set the same GPU index in each server script.

Each server exposes one GPU and uses tensor/pipeline/data parallel sizes of 1. All three models use the same BF16 KV cache, context limit, batching settings, and GPU-memory utilization. Quantization is detected from each checkpoint's configuration; the original BF16 model has none. Prefix caching is disabled so shared prompts and repeated runs do not reuse a previous run's cached prefixes. Keep these settings and the physical GPU unchanged for a controlled comparison.

All five clients (Original, FP8, W4A16, W4A16 Marlin, and NVFP4 Auto) prepare the prompts once, then benchmark **16, 64, and 256 QPS sequentially**. Each rate uses the same 1,032 requests: 516 conversations with both instruction conditions. That is 3,096 measured requests per script. Comment/uncomment the `run_benchmark 16`, `run_benchmark 64`, and `run_benchmark 256` calls to select rates. There is no client `--max-concurrency` cap. Requests can queue inside the server under load, and achieved throughput can remain below the target. Server scheduling limits are configured separately in the serving scripts.

Each run uses greedy decoding, a maximum of 256 generated tokens, and 10 warmup requests excluded from the measured run. Server startup and prompt preparation are also excluded. HTTP timing includes streaming, networking, and server work; chat-template rendering happens beforehand.

The terminal report and saved Markdown summary contain these six metrics, aggregated over both instruction conditions:

| TTFT Mean (ms) | TTFT P95 (ms) | Per Token Mean (ms/token) | Per Token P95 (ms/token) | E2E Mean (ms) | E2E P95 (ms) |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

“Per Token” is vLLM's TPOT: for each response with more than one output token, `(end-to-end latency - TTFT) / (output tokens - 1)`. The mean and P95 summarize those per-response values, excluding the first token. `report_speed.py` reads the completed vLLM result and prints the table; incomplete/failed runs produce an error instead of a summary.

E2E measures each request from the client starting the HTTP operation to receiving
the last completion chunk, including network time, server queueing, prefill, and decode.
This is vLLM's E2EL definition; trailing usage-only chunks and the SSE `[DONE]`
marker do not advance the measured end time.
It excludes startup, warmups, prompt preparation, and any client admission queue
before the HTTP operation starts. The clients have no concurrency cap.
`report_speed.py` also computes E2E Mean/P95 from the `latencies` array (seconds)
in older raw JSON files, using linear interpolation for P95, so rerunning inference
is not required. Actual output lengths can differ between models and affect E2E.

Results stay under `outputs/quantization/speed/original/`, `outputs/quantization/speed/fp8/`, `outputs/quantization/speed/w4a16/`, and `outputs/quantization/speed/w4a16_marlin/`. Each rate saves `qps<rate>_<timestamp>_summary.md`, `qps<rate>_<timestamp>_raw.json`, and `qps<rate>_<timestamp>.log`; all three rates share a timestamp. Prompt preparation messages go to `prepare_<timestamp>.log`. Full benchmark diagnostics, including achieved throughput and request counts, are retained in the raw JSON/log instead of being printed as extra report metrics. Errors on stderr remain visible.

Responses stop normally at EOS, so this measures the actual query-rewriting workload. The models may produce different output lengths. For a fixed amount of decode work, add `--ignore-eos` to each client's benchmark command, forcing exactly 256 output tokens per request. Keep that setting identical across all three models and use a separate result directory for those runs.

The prepared prompts and a checksum/count manifest are written under `outputs/quantization/speed/`. They contain test conversations and are covered by the repository's `outputs/` ignore rule. No API keys are needed for this local benchmark.

Each new benchmark uses its own `test_prompts_<timestamp>.jsonl` and matching
manifest, and saves a copy of the benchmark launcher. `--save-detailed` retains
per-request input/output token counts, TTFT, E2E, start times, and generated text
for checking output-length effects and arrival patterns. These artifacts are
local and ignored by Git. Server launchers save their own script snapshot,
package versions, GPU details, and combined stdout/stderr log under
`logs/quantization/`; inspect this log to confirm which kernel `auto` selected.

The benchmark request rates are offered load, not GPU batch sizes or achieved
throughput. The server schedules at most 64 sequences at once; excess requests
can wait in its queue. A 1,032-request run at 256 QPS offers traffic for roughly
four seconds before draining, so it is a finite overload test when the server
cannot sustain 256 requests/sec. It is not a long steady-state latency result.

## Comparing on L40S

The current workspace has one NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with 96 GB memory. L40S uses the Ada architecture, has 48GB GDDR6 and 864GB/s memory bandwidth, and supports native FP8 Tensor Core operations ([NVIDIA specifications](https://www.nvidia.com/en-gb/data-center/l40s/)). vLLM supports FP8 W8A8 on Ada and Hopper, and W4A16 through supported quantization kernels ([FP8 support](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/), [quantization compatibility](https://docs.vllm.ai/en/latest/features/quantization/)).

Absolute speed and the relative FP8/W4A16 ranking can change with memory bandwidth, kernel selection, prompt length, and concurrency. W4A16's smaller weights may help memory-bound decoding; FP8 W8A8 may benefit compute-heavy prefill and larger batches. These are possible advantages, not measured results for this model. Benchmark both formats on the actual L40S using the same vLLM version and settings; H100 timings do not establish which format wins on L40S. The existing checkpoints can be reused on supported L40S software without requantizing solely for the GPU change.
