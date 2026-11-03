# Voice-Agent Retrieval Query Rewriter

Data construction, supervised fine-tuning (SFT), direct preference optimization (DPO), and evaluation for a voice-agent retrieval query rewriter built on Qwen3-4B-Instruct-2507.

The model turns a conversation into a concise search query for knowledge needed by the agent's **next response**. It supports additional user-defined rewriting instructions, while preserving relevant names, entities, and constraints.

## Redesigned pipeline

![SFT and DPO data construction and training pipeline](figure/image.png)

1. **Combine two sources:** previous instruction data and recently exported conversations.
2. **Generate user instructions:** use Gemini-3.8 and Gemini-3.5 for self-instruction, drawing on seed instructions and conversation summaries.
3. **Generate teacher responses:** randomly select a subset from each source and use Gemini-3.8 to produce rewrites both with and without the additional user instruction. The intended budget is 200K prompt/response examples per source before quality selection.
4. **Judge and select SFT data:** use a separate judge to assess query quality without the additional instruction and instruction following with it. Filter by scores and sample the retained pairs, targeting 100K SFT examples per source.
5. **Train combined SFT:** combine the two selected sources into 200K SFT examples.
6. **Construct DPO data from instructions unused by SFT:** generate student responses with the SFT checkpoint, score and filter them, generate missing Gemini-3.8 teacher responses, and compare teacher/student responses anonymously with Gemini-4. Keep teacher-winning pairs that also satisfy the configured length constraints.
7. **Train DPO from the SFT checkpoint**, then evaluate query quality and instruction following under their corresponding input conditions.

### Counting examples

One selected conversation with two instruction conditions produces **two SFT examples**: one with the user instruction and one without it. Thus, 50K selected conversations from each source produce 100K examples per source and 200K combined. A **DPO pair** instead contains one prompt, one chosen response, and one rejected response.

These are target counts, not guarantees from score filtering. The shell files retain experiment-specific settings: for example, the previous-source construction currently has a 50K-conversation response limit, while the exported-source construction has a 100K limit. Review limits and enabled stages before reproducing the full target pipeline.

## Repository layout

| Path | Purpose |
|---|---|
| [scripts/](scripts/) | Shell entry points and experiment settings |
| [scripts/src/](scripts/src/) | Python data construction, preparation, training, and evaluation code |
| [case_study/](case_study/) | Case-study tools and curated retrieval issue cases |
| [figure/image.png](figure/image.png) | Pipeline diagram |
| [data/query_rewriter_exported_sample10k_random1k_filtered.json](data/query_rewriter_exported_sample10k_random1k_filtered.json) | Included held-out test set, currently 516 records |
| [plans/](plans/) | Earlier implementation plans; the workflow above describes the final design |

The repository excludes local credentials, large training artifacts, environments, generated outputs, and archived scripts through [.gitignore](.gitignore). The full training sources and model weights must be supplied locally.

## Setup and execution

Training requires a CUDA environment with PyTorch, Unsloth, Transformers, TRL, PEFT, and Datasets. Local inference uses vLLM; remote generation and judging use the OpenAI and Google Gen AI clients. This repository does not currently provide a pinned dependency environment.

Before running a shell script, configure its Python interpreter paths, local model/data paths, GPU allocation, and API model names. Several scripts use separate interpreters for preparation, training, and evaluation. Model identifiers in the scripts reflect the experiment environment and must be available through your configured API service.

Provide credentials through environment variables:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Keep these exports in `~/api-key.txt`, outside the repository, with permissions `600` (`chmod 600 ~/api-key.txt`). The API-using shell scripts load this file automatically when it exists. To load the keys manually, run `source "$HOME/api-key.txt"`. You can also supply the environment variables directly when the file is absent.

**Shell files are editable experiment recipes.** Enable the stages you want by uncommenting their command blocks. Some stages are deliberately commented out after earlier runs; invoking a shell does not necessarily execute the entire pipeline. Run the commands below from the repository root, in the required dependency order.

### W&B training charts

`scripts/train_dpo_combined.sh` enables W&B logging to `damian796-personal/query-rewriter-IF-training`, using the model name as the run name. Log in once from the repository root:

```bash
.venv-finetune/bin/wandb login
```

Install `wandb` in the training environment when setting up another machine. Alternatively, export `WANDB_API_KEY` in `~/api-key.txt`. Entity and project names do not replace authentication.

Both `scripts/src/train_dpo.py` and `scripts/src/train_sft.py` accept `--report-to wandb` and an optional `--run-name`. Other training shells can enable logging by passing `--report-to wandb`. Charts include training loss and learning rate, plus the reward metrics emitted by DPOTrainer. Metrics are logged every 10 optimizer steps; distributed training creates one run on the main process. Checkpoint uploads are disabled by default. The separate post-training query evaluations still write their existing local outputs, rather than W&B metrics.

Use `--report-to none` to disable tracking, or `WANDB_MODE=offline` to record locally. Each invocation creates a fresh W&B run by default, including when resuming a training checkpoint.

## Test and seed data

```bash
bash scripts/construct_test_data.sh
bash scripts/construct_seed_data.sh
```

[construct_test_data.sh](scripts/construct_test_data.sh) builds a 10K candidate pool, randomly samples 1K records, applies language and conversation-quality filtering, and labels retrieval necessity. The final test set retains both retrieval-needed and retrieval-not-needed cases. You can use the included test set without reconstructing it.

[construct_seed_data.sh](scripts/construct_seed_data.sh) builds seed training examples from exported conversations, caps records per customer, excludes test overlap, and generates responses for both instruction conditions.

## SFT construction and training

```bash
bash scripts/construct_sft_data_previous.sh
bash scripts/construct_sft_data_exported.sh
bash scripts/train_sft_combined.sh
```

The construction shells share [construct_sft_data.py](scripts/src/construct_sft_data.py). `construct-inst` generates user instructions; `construct-resp` generates their responses. Both stages use the same self-instruction file, allowing instructions to exist before teacher responses are requested. Scoring writes a separate `_scored` artifact, and score-based selection produces the `_sft` input.

The exported source additionally uses conversation deduplication and company-balanced sampling. Its current scoring model is `gpt-6-luna`; the previous-source shell uses `gemini-3.5-flash-lite`. Quality and instruction-following thresholds are configurable independently.

[prepare_sft_data.py](scripts/src/prepare_sft_data.py) excludes unusable pairs and test overlap, expands both instruction conditions into prompt/completion examples, and prepares an Arrow tokenized dataset. Preparation metadata controls cache reuse. [train_sft.py](scripts/src/train_sft.py) performs the actual training.

For separate source experiments, use [train_sft_previous.sh](scripts/train_sft_previous.sh) or [train_sft_exported.sh](scripts/train_sft_exported.sh). The combined recipe names its model `Qwen3-4B-Instruct-2507-combined_selfins-200K_LoRA`.

## DPO construction and training

```bash
bash scripts/construct_dpo_data_previous.sh
bash scripts/construct_dpo_data_exported.sh

# Train either source experiment after its comparisons finish.
bash scripts/train_dpo_previous.sh
bash scripts/train_dpo_exported.sh
```

The construction shells use [construct_dpo_data.py](scripts/src/construct_dpo_data.py):

1. **Extract:** check against the actual SFT data and select one unused user-instruction variant per conversation. If the only variant was used by SFT, drop that conversation from this selection.
2. **Generate:** produce student responses with and without that instruction using the combined SFT checkpoint.
3. **Score:** judge the student response. With instructions, score applicability and instruction following; without instructions, score general query quality.
4. **Compare:** first select eligible records using the score thresholds, then reuse completed comparisons or generate missing teacher answers and compare the two responses anonymously. With instructions, emphasize instruction following; without instructions, emphasize query quality.

The training shells perform **prepare** immediately before training. Preparation reapplies score thresholds, retains teacher-winning comparisons, applies token-length ratios and test exclusions, and writes the combined preference dataset. Gemini supplies `chosen`; the SFT model supplies `rejected`.

### Five preparation controls

| Setting | Meaning |
|---|---|
| `min_applicability_score` | Minimum applicability for the with-instruction group |
| `max_rejected_score` | Maximum student instruction-following score for that group |
| `max_without_instruction_rejected_score` | Maximum student query-quality score for the without-instruction group |
| `with_instruction_length_ratio` | Maximum teacher/student response token-length ratio with instructions |
| `without_instruction_length_ratio` | Maximum teacher/student response token-length ratio without instructions |

A ratio of `1.0` allows teacher answers no longer than the student answer; `2.0` allows up to twice its length. **Setting a group's ratio to `0.0` excludes that group.** This is a response-length constraint, not a sampling fraction.

Comparison outputs have threshold-independent names and retain cached records outside the current score range. This allows later runs to expand the range without repeating existing valid comparisons. Preparation filters that accumulated cache using the current settings. Use a separate comparison output when changing the judging experiment in ways that make earlier decisions incompatible.

[train_dpo.py](scripts/src/train_dpo.py) initializes from the finished SFT adapter and uses the initial SFT policy as its reference. The source-specific training shells configure four GPUs each and include evaluation commands after training; inspect the enabled blocks before running them.

## Evaluation

Set the model and checkpoint paths in [test_query_rewriter.sh](scripts/test_query_rewriter.sh), then run:

```bash
bash scripts/test_query_rewriter.sh
```

[test_query_rewriter.py](scripts/src/test_query_rewriter.py) generates both instruction conditions in one inference run and evaluates them under matching prompts:

| Metric | Generated response and judging context |
|---|---|
| Without-instruction quality | Response generated without the additional instruction, judged without it |
| With-instruction IF | Response generated with the additional instruction, judged with it |
| Retrieval-needed with-instruction IF | The same IF metric restricted to retrieval-needed test cases |
| Average output length | Response tokens for each condition, measured with a common tokenizer |

Scores use a 0–5 scale. The report starts with:

```text
Core Results: without-inst quality: ... | without-inst length: ... | with-inst IF: ... | with-inst length: ... | retrieval-need with-inst IF: ...
```

Detailed metrics follow. These judge scores assess the rewritten queries; they do not directly measure downstream retrieval recall.

## Additional experiments

- **Contrast DPO:** [compare_dpo_data_contrast_exported.sh](scripts/compare_dpo_data_contrast_exported.sh) and [compare_dpo_data_contrast_previous.sh](scripts/compare_dpo_data_contrast_previous.sh) pair a teacher response generated **with** the instruction against a student response generated **without** it. Comparison and training still include the user instruction. Train with [train_dpo_contrast_exported.sh](scripts/train_dpo_contrast_exported.sh) or [train_dpo_contrast_previous.sh](scripts/train_dpo_contrast_previous.sh). This is a separate ablation from ordinary DPO.
- **Case-study CSV extension:** [extend_case_study_responses.sh](scripts/extend_case_study_responses.sh) aligns cases by record ID and adds two response columns per model, one for each instruction condition.
- **Retrieval issue cases:** [case_study/test_query_rewriter.sh](case_study/test_query_rewriter.sh) runs inference on the curated case-study inputs.

Many data files use JSON Lines even when their suffix is `.json`: read one JSON object per nonempty line. Keep source records, instruction variants, generation settings, and comparison decisions aligned when reusing intermediate artifacts.
