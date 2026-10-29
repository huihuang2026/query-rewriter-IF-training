# Query-rewriter fine-tuning plan

Status: implemented in `process_data.py`, `train_sft.py`, and `train.sh`. Data
preparation has been run. Training dependencies are installed separately in
`.venv-finetune/` (Python 3.12.15); the shell uses that environment directly.
Training is launched manually. Current speed preference: CPU/embedding offloading
and gradient checkpointing are disabled. Activations stay on GPU, requiring more
VRAM; the batch size may need reducing to fit.

Current trial: `scripts/trash-bin/train.sh` uses a random 6,000-record sample (seed 42)
from 32,999 usable completed records in the current Gemini output:
`query_rewriter_training_max5_gemini_pairs_6000.json`. Preparation produced
`query_rewriter_training_max5_processed_6000.json` with 11,997 training examples
after dropping three empty queries. The earlier processed 28,999-record dataset
was deleted; its generated-pair snapshot and the live Gemini output are preserved.
LoRA checkpoints go to `ckpts/Qwen3-4B-Instruct-2507-LoRA-6000/`.
The shell retains the user's batch size 4, accumulation 4, and 8,192-token limit.
Processing is already complete and its shell command remains commented out.
The following full-data workflow describes the later complete run.

## 1. Files

Use the existing directory `scripts/`, matching its current workspace name:

```text
plans/
  FINETUNE_PLAN.md # This fine-tuning plan
  QUERY_REVIEW_PLAN.md
scripts/
  process_data.py  # Validate and convert Gemini pairs into training examples
  train_sft.py         # Unsloth-based supervised fine-tuning
  train.sh         # Run processing, then training
```

Use installed Unsloth and Hugging Face libraries. Leave Unsloth's package source
and the older `unsloth-finetune/` classifier project unchanged.

## 2. Input and filenames

Read the generated Gemini pairs, not the sampled source file:

```text
data/export-data-train/query_rewriter_training_max5_gemini_pairs.json
```

This is the existing `query_rewriter_training_max5_gemini_pairs_NULL.json` file,
with `_NULL` removed. Use the actual repository spelling: `query_rewriter`,
`max5`, and `gemini_pairs`.

Generation filenames omit the limit suffix for unlimited runs (`NULL`, omitted,
or `0`). Numeric limits retain their suffix, such as `_gemini_pairs_200.json`.
Do not automatically combine source files, trial runs, and completed generation.

Save processed JSON Lines beside the generated input:

```text
data/export-data-train/query_rewriter_training_max5_processed.json
```

Accept JSON Lines despite the `.json` extension. Also support JSON arrays such
as `training_samples_5.json` for small preparation checks.

## 3. Data processing

Read `system_prompt`, `user_content`, `query_with_user_instruction`, and
`query_without_user_instruction`. Support flat records and the existing wrapped
`source` format. Skip generation failures, invalid input fields, and malformed
instruction sections. Preserve all languages and conversation text.

Split the original system prompt at the exact marker
`Additional instruction from the user:` and create up to two training examples:

| Example | System message | User message | Assistant target |
|---|---|---|---|
| With user instruction | Original full `system_prompt` | Original `user_content` | `query_with_user_instruction` |
| Without user instruction | Exact prefix before the marker | Same `user_content` | `query_without_user_instruction` |

Keep the full prompt unchanged. For the base variant, remove only the marker and
the additional instruction following it; preserve the prefix's existing text and
whitespace. Do not append instructions, output conventions, extra lines, or
reasoning hints. Do not modify the source files.

Treat targets separately. Remove each example whose query is missing, null,
empty, whitespace-only, or `[NO_QUERY]`. Keep the other variant if its target is
valid and the source record is otherwise valid. Do not substitute a default
query or use the original `rewritten_query` as a target.

**Do not deduplicate** source records, instructions, or prepared examples. Keep
repeated examples. Earlier sampling and test-instruction exclusion remain separate
steps; this processor does not repeat them.

Use conversational prompt/completion records:

```json
{
  "prompt": [
    {"role": "system", "content": "Unchanged full instruction or exact base prefix"},
    {"role": "user", "content": "Original multi-turn voice conversation"}
  ],
  "completion": [
    {"role": "assistant", "content": "Matching nonempty Gemini query"}
  ]
}
```

Keep embedded `agent:`/`customer:` lines inside one user-message string, matching
online inference. Do not expose metadata, source queries, or previous scores to
the model. Print input counts, invalid records, removed empty-query examples, and
written examples; no separate review reports.

## 4. Training

Use this local base model by default:

```text
/home/huanghui/WorkSpace/models/Qwen3-4B-Instruct-2507
```

Read the processed file directly and use all retained examples for training.
Train for **three epochs** with **no validation split or evaluation loop**.
Save training checkpoints and the final artifact. Testing stays in a separate
script; do not run API-based judging during training.

Use the model's supported chat template and configure non-thinking formatting
through template options for hybrid models, without altering system-message text.
Use TRL `SFTTrainer` with ordinary supervised loss on all assistant query tokens
and the ending token. Mask system and user tokens. Do not reuse the old classifier's
custom KL-divergence loss.

Default the combined prompt/target limit to 32,768 tokens. Check lengths after
formatting and report/skip oversized examples instead of silently truncating
instructions or targets. Start with sequence packing disabled.

| Mode | Behavior |
|---|---|
| `--mode lora` (default) | Freeze base weights and train LoRA adapters; initial rank 16 and alpha 32. |
| `--mode lora --load-in-4bit` | Optional QLoRA over a 4-bit-loaded base model. |
| `--mode full` | Full-parameter fine-tuning without adapters or 4-bit loading. |

Reject full tuning combined with 4-bit loading. Print trainable/total parameters.
Expose model path/ID, batch size, gradient accumulation, learning rate, sequence
length, LoRA settings, seed, and checkpoint resume as command-line options.

Initial settings: batch size 1, gradient accumulation 16, epochs 3, seed 42;
learning rate `2e-4` for LoRA or `2e-5` for full tuning, explicitly overridable.
Use BF16 where supported. Gradient checkpointing is disabled to favor speed. No external experiment
tracker by default. These are starting settings, not validated hyperparameters.

The initial model is 4B. GPU memory must still accommodate the selected training
mode and sequence length. Full tuning updates all base-model parameters. Do not
silently quantize or switch training modes if memory is insufficient.

## 5. One simple shell script

Keep settings near the top and run the two Python scripts sequentially. Stop if
processing fails. Intended structure:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

data_dir="data/export-data-train"
input_file="$data_dir/query_rewriter_training_max5_gemini_pairs.json"
processed_file="$data_dir/query_rewriter_training_max5_processed.json"
model_name="Qwen3-4B-Instruct-2507"
model="/home/huanghui/WorkSpace/models/${model_name}"
suffix="" # Optional settings suffix, e.g. -epochs3.

python scripts/src/trash-bin/process_data.py --input "$input_file" --output "$processed_file"

# LoRA training (enabled).
python scripts/src/train_sft.py \
    --input "$processed_file" --output-dir "ckpts/${model_name}-LoRA${suffix}" \
    --model "$model" --mode lora --epochs 3 "$@"

# Full-parameter / non-LoRA training (disabled).
# python scripts/src/train_sft.py \
#     --input "$processed_file" --output-dir "ckpts/${model_name}${suffix}" \
#     --model "$model" --mode full --epochs 3 "$@"
```

Data processing always reads the entire input file; there is no processing
`--limit` option. To switch to full-parameter training, comment out the LoRA
command and uncomment the full-training command. The modes have separate output
directories. An optional training `--max-steps` supports a short GPU check before
the full run.

Run directories are directly under `ckpts/`, with the model name first and
fine-tuning settings as suffixes: `Qwen3-4B-Instruct-2507-LoRA` for LoRA, or
`Qwen3-4B-Instruct-2507` for full tuning. Append optional settings using the shell's
`suffix` variable. Save the tokenizer and model artifacts under this run directory.
LoRA saves adapters
and their base-model reference; full tuning saves updated weights. Merged LoRA
export is optional because large-model merging needs additional resources.

## 6. Verification

1. Correct full/base target mapping and exact preservation of system text.
2. Empty queries and invalid records removed; duplicates retained.
3. Processed file saved beside the input with the required filename.
4. Only assistant targets contribute to loss.
5. Three epochs by default, without a validation dataset or evaluation loop.
6. Correct LoRA/full parameter selection and reloadable/resumable checkpoints.

Installed versions in `.venv-finetune/`: Unsloth/Unsloth Zoo 2026.10.5,
PyTorch 2.14.1+cu132, Transformers 5.17.0, TRL 1.15.0, Datasets 4.8.5,
and PEFT 0.21.2. The base environment is unchanged. Installation followed the
[Unsloth Core guide](https://unsloth.ai/docs/get-started/install/pip-install).

Run both stages with:

```bash
bash scripts/trash-bin/train.sh
```

Run data preparation alone with:

```bash
python scripts/src/trash-bin/process_data.py
```

The trainer creates explicit token labels with prompt tokens set to `-100`,
including the checkpoint's non-thinking generation prefix. Its custom label
collator preserves those masks, so TRL's automatic completion/assistant masking
is disabled to avoid applying a second, template-dependent mask.

## References

- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Unsloth multi-GPU training](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)
