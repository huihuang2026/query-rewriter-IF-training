# Quantization and speed comparison

All scripts use the existing train02 checkpoint: 700K SFT + previous DPO/length control + IF4/CP4. Edit variables directly and comment/uncomment command blocks.

## Quantization

- `quantize_w4a16.sh`: merge the BF16 base and train02 LoRA, then run GPTQ W4A16 with the saved 10,240 calibration prompts. Output suffix: `-W4A16-GPTQ-calib10240`.
- `quantize_fp8.sh`: merge the same base/adapter and apply data-free `FP8_DYNAMIC`, then optionally generate, judge, and report quality metrics. Output suffix: `-FP8-DYNAMIC`.

Quantization uses `.venv-quantization/bin/python`. All `serve_*.sh` and
`benchmark_*.sh` scripts use the project-local `.venv/bin/python` and add its
command directory to `PATH` for runtime compilation tools such as `ninja`.
The evaluation steps in `quantize_fp8.sh` also use `.venv/bin/python`.
Create it with Python 3.12 and install the serving dependencies:

```bash
sudo apt-get install -y python3.12-dev build-essential
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r quantization/requirements-vllm.txt
```

All server/client pairs load their checkpoints and bundled tokenizers from `ckpts/`.
Re-running FP8/W4A16 quantization or the optional BF16 merge requires the original
unmerged base model at `models/Qwen3-4B-Instruct-2507` and the trained adapter under `ckpts/`.
All servers use GPU 0 on this machine. Run one server at a time: stop the previous
server before starting another, since each reserves 90% of GPU memory.

The `vllm[bench]` requirement also installs the benchmark dependencies, including `pandas`, `matplotlib`, `seaborn`, `datasets`, `scipy`, and `plotly`.

## NVFP4

Edit the paths and sample count in `quantize_nvfp4.sh`, then run
`bash quantization/quantize_nvfp4.sh`. It calls the Python quantizer described below.

`quantize_nvfp4.py` loads an already merged BF16 checkpoint (by default the
train02 `-BF16` model in `ckpts/`) and applies LLM Compressor's standard
`QuantizationModifier(scheme="NVFP4")` recipe. This is **W4A4 floating point**,
with groups of 16; it differs from the GPTQ INT4 W4A16 recipe. `lm_head` stays
BF16 and KV-cache quantization is not enabled. The result is saved in
compressed-tensors format alongside the tokenizer and a quantization manifest.
The default output is `ckpts/<train02-name>-NVFP4`; existing outputs are refused.

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

Run one benchmark command for each format. It starts the matching server, waits
for `/health` and the expected model in `/v1/models`, runs QPS 16 and 64, and stops
the server afterward. The serving scripts remain available for standalone use.

| Format | Standalone server | Start server and benchmark | Port |
|---|---|---|---|
| Original BF16 | `serve_original.sh` | `benchmark_original.sh` | 8002 |
| FP8 | `serve_fp8.sh` | `benchmark_fp8.sh` | 8001 |
| W4A16 Marlin | `serve_w4a16.sh` | `benchmark_w4a16.sh` | 8000 |
| NVFP4 Auto | `serve_nvfp4.sh` | `benchmark_nvfp4.sh` | 8004 |

After NVFP4 quantization completes, run `bash quantization/benchmark_nvfp4.sh`.
The server selects `--linear-backend auto` and loads
`ckpts/<train02-name>-NVFP4`. Results are saved under
`outputs/quantization/speed/nvfp4_auto/`.

`serve_original.sh` serves the existing merged BF16 model at `ckpts/<train02-name>-BF16`. Its optional merge command is commented out; enable it only when the original base model, adapter, and quantization environment are available. The helper `merge_original.py` uses the same BF16 base/adapter merge procedure as the two quantizers. This makes the baseline a full model and avoids runtime LoRA overhead affecting the comparison. All server scripts set `CUDA_VISIBLE_DEVICES=0`; edit it for another machine.

They do not call an external judge or recalculate quality scores. The benchmark renders the same system/user prompts as the existing evaluation, with thinking disabled, and sends them to the local `/v1/completions` endpoint. `prepare_speed_data.py` handles that conversion; the timing and throughput measurements use [vLLM's benchmark client](https://docs.vllm.ai/en/v0.31.0/benchmarking/cli/).

For example, benchmark FP8 from one terminal:

```bash
cd /home/huanghui/query-rewriter-IF-training
bash quantization/benchmark_fp8.sh
```

Use `benchmark_original.sh`, `benchmark_w4a16.sh`, or `benchmark_nvfp4.sh` for the
other formats. Run them sequentially on the single GPU. Model loading and CUDA
Graph capture can take several minutes; the launcher prints the server log path
and allows up to 20 minutes for readiness. It reports startup failures and stops
its own server and client processes on completion, errors, or Ctrl+C. An occupied
port is reported as an error; an existing server is not stopped or reused.

To benchmark a server you started separately with `serve_*.sh`, pass
`--client-only` to the matching benchmark script. This mode leaves that server
running. The default command needs no separate serving step.

Each server exposes one GPU and uses tensor/pipeline/data parallel sizes of 1. All four models use a BF16 KV cache. Quantization is detected from each checkpoint's configuration; the original BF16 model has none. Scheduler, CUDA Graph, and prefix-cache settings are configured in each serving script. To isolate quantization effects, hold those settings and the physical GPU fixed.

### B300 serving defaults and measured results (2026-11-17)

All four servers use 64 sequences, 8192 batched tokens, a CUDA Graph capture
ceiling of 8192, and **prefix caching enabled**. W4A16 explicitly uses Marlin.
The benchmark clients isolate warmup and measurement cache namespaces for every
format, allowing prefix reuse only within each measured run.

The historical measurements below used prefix caching only for the final Marlin
configuration; BF16, FP8, and NVFP4 used caching disabled. They precede the request
to enable caching for all formats and are not an all-cache-enabled comparison.
On the same 1,032 prompts, three repeats per rate with normal EOS produced these
mean end-to-end latencies. Values are in milliseconds; before values are fresh
baselines from that round. The new all-cache-enabled defaults have not been
rebenchmarked together.

| Model | QPS 16, before → after | QPS 64, before → after |
|---|---:|---:|
| Original BF16 | 80.75 → 61.47 | 159.54 → 86.73 |
| FP8, unchanged | 57.81 → 57.81 | 74.76 → 74.76 |
| W4A16 Marlin | 83.78 → 67.33 | 372.67 → 100.28 |
| NVFP4 Auto, unchanged | 49.86 → 49.86 | 65.89 → 65.89 |

Fixed 16-token controls confirm the BF16 and Marlin gains. Marlin reused 62.58%
of input tokens within each run, starting cold after warmup; its benefit depends
on shared prefixes in the workload. These are tuned per-format results, with
different cache policies. Earlier W4A16 results using Humming are a separate
baseline and must not be labeled Marlin.

Both applied changes matched all 16 cold output probes exactly. Warm prefix reuse
changed 4 of 16 Marlin probe texts, and full batched outputs vary with scheduling.
The external quality evaluator was not run, so these checks do not establish
task-quality equivalence. Detailed mean/P95 metrics, repeats, rejected candidates,
and validation are retained locally in `investigation/RCA.md` (Round 2) and
`investigation/results/round2/`; these investigation artifacts remain ignored.

### Benchmark procedure and results

All four clients (Original, FP8, W4A16 Marlin, and NVFP4 Auto) prepare the prompts once, then benchmark **16 and 64 QPS sequentially**. Each rate uses the same 1,032 requests: 516 conversations with both instruction conditions. That is 2,064 measured requests per script. Comment/uncomment the `run_benchmark 16` and `run_benchmark 64` calls to select rates. There is no client `--max-concurrency` cap. Requests can queue inside the server under load, and achieved throughput can remain below the target. Server scheduling limits are configured separately in the serving scripts. W4A16 explicitly selects Marlin; the separate `_marlin` launchers have been removed.

Each run uses greedy decoding, a maximum of 256 generated tokens, and 10 warmup requests excluded from the measured run. Server startup and prompt preparation are also excluded. HTTP timing includes streaming, networking, and server work; chat-template rendering happens beforehand.

The clients call `benchmark_speed.py`, which delegates scheduling, streaming,
timing, and metrics to the pinned vLLM benchmark. Each invocation uses a fresh
`cache_salt`, with separate namespaces for warmup and measurement. With prefix
caching enabled, measured requests can reuse prefixes from earlier requests in
the same run, but not from warmups, earlier rates, or previous benchmark runs.
This requires no development endpoints on the serving process. The raw JSON
records `prefix_cache_policy` and warmup/measurement request counts, and the
client helper is copied alongside the benchmark launcher for reproducibility.

The terminal report and saved Markdown summary contain these six metrics, aggregated over both instruction conditions and displayed to two decimal places:

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

Results stay under `outputs/quantization/speed/original/`, `outputs/quantization/speed/fp8/`, `outputs/quantization/speed/w4a16/`, and `outputs/quantization/speed/nvfp4_auto/`. Each rate saves `qps<rate>_<timestamp>_summary.md`, `qps<rate>_<timestamp>_raw.json`, and `qps<rate>_<timestamp>.log`; both rates share a timestamp. Prompt preparation messages go to `prepare_<timestamp>.log`. Full benchmark diagnostics, including achieved throughput and request counts, are retained in the raw JSON/log instead of being printed as extra report metrics. Errors on stderr remain visible. Older `w4a16/` results may use Humming; inspect their saved server log rather than inferring the backend from the directory name.

Responses stop normally at EOS, so this measures the actual query-rewriting workload. The models may produce different output lengths. For a fixed amount of decode work, set `max_new_tokens=16` and add `--ignore-eos` to each client's benchmark command, forcing exactly 16 output tokens per request. Keep that setting identical across all four models and use a separate result directory for those runs.

The prepared prompts and a checksum/count manifest are written under `outputs/quantization/speed/`. They contain test conversations and are covered by the repository's `outputs/` ignore rule. No API keys are needed for this local benchmark.

Each new benchmark uses its own `test_prompts_<timestamp>.jsonl` and matching
manifest, and saves copies of the benchmark launcher, client helper, and server
lifecycle helper. Automatically started runs also save `server_<timestamp>.txt`
with the server log path. `--save-detailed` retains
per-request input/output token counts, TTFT, E2E, start times, and generated text
for checking output-length effects and arrival patterns. These artifacts are
local and ignored by Git. Server launchers save their own script snapshot,
package versions, GPU details, and combined stdout/stderr log under
`logs/quantization/`; inspect this log to confirm which kernel `auto` selected.

The benchmark request rates are offered load, not GPU batch sizes or achieved
throughput. The server schedules at most 64 sequences at once; excess requests
can wait in its queue. A 1,032-request run offers traffic for roughly 65 seconds
at 16 QPS or 16 seconds at 64 QPS before draining. These are finite-load tests,
not long steady-state latency results.

## Comparing on L40S

The current workspace has one NVIDIA B300; the earlier RTX PRO 6000 measurements describe a different machine. L40S uses the Ada architecture, has 48GB GDDR6 and 864GB/s memory bandwidth, and supports native FP8 Tensor Core operations ([NVIDIA specifications](https://www.nvidia.com/en-gb/data-center/l40s/)). vLLM supports FP8 W8A8 on Ada and Hopper, and W4A16 through supported quantization kernels ([FP8 support](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/), [quantization compatibility](https://docs.vllm.ai/en/latest/features/quantization/)).

Absolute speed and the relative FP8/W4A16 ranking can change with memory bandwidth, kernel selection, prompt length, and concurrency. W4A16's smaller weights may help memory-bound decoding; FP8 W8A8 may benefit compute-heavy prefill and larger batches. These are possible advantages, not measured results for this model. Benchmark both formats on the actual L40S using the same vLLM version and settings; H100 timings do not establish which format wins on L40S. The existing checkpoints can be reused on supported L40S software without requantizing solely for the GPU change.
