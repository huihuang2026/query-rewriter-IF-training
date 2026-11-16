#!/usr/bin/env bash
# Teacher-corrected DPO. Comment/uncomment the steps you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# set -a
# source .envs.apikeys
# set +a

export CUDA_VISIBLE_DEVICES=0
python_bin=python
prepare_python="$PWD/.venv-finetune/bin/python"
base_model="models/Qwen3-4B-Instruct-2507"
student_name="Qwen3-4B-Instruct-2507-DPO"
student_adapter="ckpts/$student_name/final"
input_file="data/query_rewriter_combined_selfins_unused_instructions.json"
prefix="data/query_rewriter_dpo_contrast"
sampled_file="${prefix}_source20K_seed42.jsonl"
with_prefix="${prefix}_with_user_instruction"
no_prefix="${prefix}_without_user_instruction"
output_file="${prefix}_train.jsonl"
judge_model="claude-sonnet-5-5"
teacher_model="claude-sonnet-5-5"
conversation_limit=20000
num_gpus=1
max_instruction_following=4
max_conciseness=4
no_inst_length_ratio=1.0
seed=42

# 1. Sample the same conversations for both conditions. Comment out after the first run.
# Choose a new sampled_file when changing the count, seed, or source.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py sample-conversations \
    --input "$input_file" --output "$sampled_file" --count "$conversation_limit" --seed "$seed"

# 2. Generate student rewrites WITH user instructions.
"$python_bin" -u scripts/src/construct_dpo_data.py generate \
    --input "$sampled_file" --output "${with_prefix}_generated.jsonl" \
    --model "$base_model" --sft-adapter "$student_adapter" --model-name "$student_name" \
    --instruction-mode with_user_instruction --conversation-limit NULL --seed "$seed" --limit NULL \
    --max-model-len 4096 --max-new-tokens 512 --num-gpus "$num_gpus" --tensor-parallel-size 1

# 3. Score WITH-instruction answers.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py score \
    --input "${with_prefix}_generated.jsonl" --output "${with_prefix}_scored.jsonl" \
    --model "$judge_model" --provider anthropic --mode with_user_instruction \
    --thinking-level medium --batch-size 100 --workers 20 --limit NULL

# 4. Select IF < 4 OR compactness < 4.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py select \
    --input "${with_prefix}_scored.jsonl" --output "${with_prefix}_selected.jsonl" \
    --mode with_user_instruction --max-instruction-following "$max_instruction_following" \
    --max-conciseness "$max_conciseness"

# 5. Ask the teacher to correct the selected WITH-instruction answers.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py correct-positive \
    --input "${with_prefix}_selected.jsonl" --output "${with_prefix}_corrected.jsonl" \
    --model "$teacher_model" --provider anthropic --mode with_user_instruction \
    --max-instruction-following "$max_instruction_following" --max-conciseness "$max_conciseness" \
    --thinking-level low --batch-size 100 --workers 20 --limit NULL

# 6. Prepare WITH-instruction pairs, with both branches fitting 4096 tokens.
"$prepare_python" -u scripts/src/construct_dpo_data_contrast.py prepare \
    --input "${with_prefix}_corrected.jsonl" --selected "${with_prefix}_selected.jsonl" \
    --output "${with_prefix}_train.jsonl" --mode with_user_instruction \
    --tokenizer "$base_model" --max-seq-length 4096

# 7. Generate student rewrites WITHOUT user instructions.
"$python_bin" -u scripts/src/construct_dpo_data.py generate \
    --input "$sampled_file" --output "${no_prefix}_generated.jsonl" \
    --model "$base_model" --sft-adapter "$student_adapter" --model-name "$student_name" \
    --instruction-mode without_user_instruction --conversation-limit NULL --seed "$seed" --limit NULL \
    --max-model-len 4096 --max-new-tokens 512 --num-gpus "$num_gpus" --tensor-parallel-size 1

# 8. Score WITHOUT-instruction answers.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py score \
    --input "${no_prefix}_generated.jsonl" --output "${no_prefix}_scored.jsonl" \
    --model "$judge_model" --provider anthropic --mode without_user_instruction \
    --thinking-level medium --batch-size 100 --workers 20 --limit NULL

# 9. Select poor WITHOUT-instruction answers.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py select \
    --input "${no_prefix}_scored.jsonl" --output "${no_prefix}_selected.jsonl" \
    --mode without_user_instruction --max-instruction-following "$max_instruction_following" \
    --max-conciseness "$max_conciseness"

# 10. Ask the teacher to correct them.
"$python_bin" -u scripts/src/construct_dpo_data_contrast.py correct-positive \
    --input "${no_prefix}_selected.jsonl" --output "${no_prefix}_corrected.jsonl" \
    --model "$teacher_model" --provider anthropic --mode without_user_instruction \
    --max-instruction-following "$max_instruction_following" --max-conciseness "$max_conciseness" \
    --thinking-level low --batch-size 100 --workers 20 --limit NULL

# 11. Prepare WITHOUT-instruction pairs, requiring corrected answers no longer than negatives.
"$prepare_python" -u scripts/src/construct_dpo_data_contrast.py prepare \
    --input "${no_prefix}_corrected.jsonl" --selected "${no_prefix}_selected.jsonl" \
    --output "${no_prefix}_train.jsonl" --mode without_user_instruction \
    --tokenizer "$base_model" --max-seq-length 4096 --max-response-length-ratio "$no_inst_length_ratio"

# 12. Combine the two conditions into the final training file.
"$python_bin" -u scripts/src/merge_jsonl.py \
    --input "${with_prefix}_train.jsonl" "${no_prefix}_train.jsonl" --output "$output_file"
