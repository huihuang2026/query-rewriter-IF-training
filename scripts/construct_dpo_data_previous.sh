#!/usr/bin/env bash
# Run after combined 200K SFT training; this shell selects candidates from its own source.
# Comment individual command blocks to run each stage manually.
export CUDA_VISIBLE_DEVICES=0,1,2,3
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/construct_dpo_data_previous.sh"
    echo "Stages: extract, generate, score, compare (includes filtering). Edit settings or comment stages below."
    echo "Extraction accepts instructions without teacher responses and excludes actual SFT instruction pairs."
    echo "Comparison applies current score thresholds and reuses cached results across threshold changes."
    echo "Missing teacher answers are generated only for eligible unfinished comparisons. Prepare/train: train_dpo_previous.sh."
    exit 0
fi

if [[ -f "$HOME/api-key.txt" ]]; then
    source "$HOME/api-key.txt"
fi

python_bin="$HOME/miniconda3/bin/python"
prepare_python="$PWD/.venv-finetune/bin/python"
data_dir="data"
source_data="$data_dir/query_rewriter_previous_selfins.json"
# Exclude instructions actually used in combined SFT, including sequence-length exclusions.
# Match the dataset and adapter configured in train_sft_combined.sh.
sft_data="$data_dir/query_rewriter_combined_selfins_tokenized"
base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
sft_model_name="${base_model}-combined_selfins-200K_LoRA"
sft_adapter="ckpts/$sft_model_name/final"
test_file="$data_dir/query_rewriter_exported_sample10k_random1k_filtered.json"
num_gpus=4
seed=42
limit="NULL"
judge_model="gemini-3.5-flash-lite"
teacher_model="gemini-3.8-flash"
compare_model="gemini-4-argon"
min_applicability_score=1
max_rejected_score=4 # Instruction following with user instructions.
max_without_instruction_rejected_score=4 # Query quality without user instructions.
prefix="$data_dir/query_rewriter_previous_selfins"
unused_file="${prefix}_unused_instructions.json"
generated_file="${prefix}_unused_generated.json"
scored_file="${prefix}_unused_scored.json"
# Shared comparison cache across score thresholds; final selection is in train_dpo_previous.sh.
compared_file="${prefix}_compared.json"
batch_size=500
workers=50
max_seq_length=4096
max_new_tokens=512
mkdir -p "$data_dir"

# # 1. Select one unused instruction per conversation, including single-variant inputs.
# # If its only instruction was used in SFT, drop that conversation.
# # The source contains instructions and any completed teacher responses together.
# "$prepare_python" -u scripts/src/construct_dpo_data.py extract \
#     --input "$source_data" --sft-data "$sft_data" --tokenizer "$model" --test "$test_file" \
#     --allow-missing-responses \
#     --output "$unused_file" --seed "$seed" --limit "$limit"

# # 2. Generate SFT responses with and without the selected user instruction.
# "$python_bin" -u scripts/src/construct_dpo_data.py generate \
#     --input "$unused_file" --output "$generated_file" \
#     --model "$model" --sft-adapter "$sft_adapter" --model-name "$sft_model_name" \
#     --num-gpus "$num_gpus" --tensor-parallel-size 1 \
#     --max-model-len "$max_seq_length" --max-new-tokens "$max_new_tokens" --seed "$seed" --limit "$limit"

# # 3. Score only SFT responses: applicability/IF with instructions, quality without.
# "$python_bin" -u scripts/src/construct_dpo_data.py score \
#     --input "$generated_file" --output "$scored_file" \
#     --judge-model "$judge_model" --batch-size "$batch_size" --workers "$workers" --limit "$limit"

# 4. Select by current score thresholds, reuse completed comparisons, then compare pending pairs.
# Generate missing Gemini 3.8 Flash answers only for selected pending pairs.
# Keep cached comparisons outside the current thresholds for future runs; report both groups.
"$python_bin" -u scripts/src/construct_dpo_data.py compare \
    --input "$scored_file" --output "$compared_file" \
    --judge-model "$compare_model" --teacher-model "$teacher_model" \
    --min-applicability-score "$min_applicability_score" --max-rejected-score "$max_rejected_score" \
    --max-without-instruction-rejected-score "$max_without_instruction_rejected_score" \
    --batch-size "$batch_size" --workers "$workers" --seed "$seed" --limit "$limit"
