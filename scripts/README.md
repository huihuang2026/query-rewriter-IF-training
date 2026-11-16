# Shell scripts

Edit the variables at the top of each shell, then comment/uncomment the numbered command blocks you want to run. There are no stage arguments or automatic input/model selection. Run with `bash scripts/<name>.sh` from the project root. Each shell also changes to the project root itself.

| Script | Steps |
|---|---|
| `construct_test_data.sh` | Sample conversations → language filter → quality judging → retrieval judging → export test set |
| `construct_seed_data.sh` | Select authentic customer instructions and generate teacher rewrites → process seed examples |
| `construct_sft_data.sh` | Prepare exported source → self-instructions for both sources → teacher rewrites → judge/rejudge → select pairs → tokenize → extract unused instructions |
| `construct_dpo_data.sh` | Generate SFT-student answers → score → compare with teacher → prepare ordinary DPO pairs |
| `construct_dpo_data_contrast.sh` | Sample conversations → generate/score/select/correct each instruction condition → prepare and combine DPO pairs |
| `train_sft.sh` | Train the SFT adapter |
| `train_dpo.sh` | Block 1: ordinary DPO; block 2: corrected-positive DPO (commented by default) |
| `test_query_rewriter.sh` | Generate both instruction conditions → judge → calculate metrics |
| `test_quantization.sh` | The same three evaluation steps using merged GPTQ weights |

Earlier experiment shells are in `trash-bin02/`.

## Settings

- Set `python_bin` to your API/vLLM environment. Set `prepare_python` to an environment with Transformers and Datasets. Training defaults to `.venv-finetune/bin/python`; quantization uses `.venv-quantization/bin/python`.
- Uncomment the API-key loading lines to load `.envs.apikeys`, or export the required Gemini, OpenAI, and Anthropic API keys yourself.
- Set `CUDA_VISIBLE_DEVICES` and `num_gpus` explicitly for your other machine.
- Edit input paths, API model names, sample counts, thresholds, and model names directly in each script. No scripts were run as part of this cleanup.

## Training data and checkpoints

Training defaults use the retained datasets directly under `data/`:

- SFT: `query_rewriter_self_instruct_summary_processed.json`.
- Ordinary DPO: `query_rewriter_self_instruct_1_4_with_noinst_length_dpo_train.json`.
- Corrected-positive DPO: `query_rewriter_6kSFT_withinst_source20K_4_4_noinst_source20K_4_4_len1.0_dpo_train.json`.

Commented input assignments show where to use newly constructed datasets instead. SFT construction creates `query_rewriter_sft_tokenized/`; the DPO constructors create `query_rewriter_dpo_train.jsonl` and `query_rewriter_dpo_contrast_train.jsonl`.

The model sequence is:

```text
models/Qwen3-4B-Instruct-2507
  -> ckpts/Qwen3-4B-Instruct-2507-SFT/final
  -> ckpts/Qwen3-4B-Instruct-2507-DPO/final
  -> ckpts/Qwen3-4B-Instruct-2507-DPO-contrast/final
```

In `train_dpo.sh`, block 1 continues SFT with ordinary DPO (SFT loss weight 0.0). Comment out block 1 and uncomment block 2 to continue DPO with corrected-positive pairs (SFT loss weight 1.0). Training and evaluation are separate commands.

## Construction notes

SFT defaults generate two instruction variants per conversation, judge one complete variant, and select 50K conversation pairs per source after quality ≥ 4 and instruction following ≥ 3 filtering. Each selected pair produces two SFT examples. Lower `sft_pairs_per_source` if fewer pairs qualify, or use `NULL` for all qualifying pairs. Changing `scoring_model` rejudges the answers; rerun selection and preparation afterward.

The SFT shell combines full generation journals and extracts unused instructions for DPO/calibration. Its merge command preserves the first source's variants when conversations overlap. The small `src/merge_jsonl.py` helper keeps merging logic out of the shells.

Ordinary DPO keeps teacher-winning pairs and applies chosen/rejected response-token ratios of at most 2.0 with instructions and 1.0 without. Contrast DPO selects instruction following < 4 or compactness < 4 and asks the teacher to correct the student answer. Without-instruction corrections must be no longer than their negatives. Both DPO branches must fit 4096 tokens.

Contrast sampling creates a shared 20K-conversation file. Comment out its sampling step once that file exists; choose a new filename when changing the cohort.

## Quantization and evaluation

Quantization and evaluation target the existing `Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch` checkpoint directly under `ckpts/`. Its original remains in `ckpts/final-round/`; the existing `-W4A16-GPTQ` model has been moved directly under `ckpts/`. The clean names above are for future training runs.

`quantization/quantize.sh` has two independent blocks: sample balanced calibration prompts, then merge/quantize the adapter. The 10,240 prompts (5,120 per instruction condition) are already saved to `data/quantization/calibration_10240_seed42.jsonl`, so sampling is commented out. Uncomment it to resample. The new model is written alongside the existing train02 checkpoint directly under `ckpts/`, with suffix `-W4A16-GPTQ-calib10240`, matching `test_quantization.sh`. The calibration limit remains 4096 tokens.

Both evaluation shells generate with and without user-defined instructions, judge with `gpt-6-astra`, and save metrics under `outputs/` (`outputs/quantization/` for GPTQ). `test_query_rewriter.sh` evaluates the original train02 adapter; `test_quantization.sh` evaluates its new 10,240-prompt GPTQ version.

For FP8, run `bash quantization/quantize_fp8.sh`. It merges the same original BF16 base and train02 adapter, then saves `ckpts/<train02-name>-FP8-DYNAMIC`. The scheme uses FP8 weights with static per-channel scales and FP8 activations with dynamic per-token scales; embeddings and `lm_head` remain BF16, and KV cache quantization is not enabled. This data-free scheme needs neither calibration prompts nor a calibration sequence-length cap. It uses the existing quantization environment. See the [vLLM FP8 guide](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/).

`quantize_fp8.sh` includes four command blocks: quantize, generate both instruction conditions, load `.envs.apikeys` and judge, then calculate metrics. Evaluation uses `test_python` (the vLLM/OpenAI environment) and saves FP8-specific results under `outputs/quantization/`. Edit the interpreter path for your machine. Comment out the quantization command once the FP8 checkpoint exists to rerun evaluation alone; comment out generation as well to rerun only scoring and metrics.

For speed only, use `quantization/serve_fp8.sh` with `quantization/benchmark_fp8.sh`, or `quantization/serve_w4a16.sh` with `quantization/benchmark_w4a16.sh`, in separate terminals. Each client measures latency and throughput on both test-set instruction conditions without judging quality. Run the formats one at a time on the same GPU. See [quantization/README.md](../quantization/README.md) for interpreting the results and comparing on L40S.

The original train02 BF16 speed baseline uses `quantization/serve_original.sh` and `quantization/benchmark_original.sh` on port 8002. The server shell includes a one-time merge of the original base and trained adapter; comment that step out once its merged checkpoint exists. Benchmark results go under `outputs/quantization/speed/original/`.

All four speed clients (Original, FP8, W4A16, and W4A16 Marlin) run 16, 64, and 256 requests/sec sequentially without a client concurrency cap. Each rate measures the same 1,032 requests, with 10 warmup requests excluded, and saves its own log, raw JSON, and Markdown summary. Comment/uncomment the three `run_benchmark` calls to select rates. Server scheduling limits remain configured in the serving scripts. The report contains only mean/P95 TTFT and mean/P95 time per output token.
