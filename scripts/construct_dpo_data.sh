#!/usr/bin/env bash
# Ordinary DPO. Comment/uncomment the steps you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# set -a
# source .envs.apikeys
# set +a

export CUDA_VISIBLE_DEVICES=0
python_bin=python
prepare_python="$PWD/.venv-finetune/bin/python"
base_model="models/Qwen3-4B-Instruct-2507"
student_name="Qwen3-4B-Instruct-2507-SFT"
student_adapter="ckpts/$student_name/final"
input_file="data/query_rewriter_combined_selfins_unused_instructions.json"
source_file="data/query_rewriter_combined_selfins.jsonl"
sft_file="data/query_rewriter_sft_processed.jsonl"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
prefix="data/query_rewriter_dpo"
output_file="${prefix}_train.jsonl"
judge_model="gemini-4-argon"
teacher_model="gemini-3.8-flash"
conversation_limit=20000
num_gpus=1
min_applicability=1
max_rejected_score=4
max_noinst_rejected_score=4
with_inst_length_ratio=2.0
no_inst_length_ratio=1.0
seed=42

# 1. Extract unused instructions. Leave commented when SFT step 8 already produced them.
# "$prepare_python" -u scripts/src/construct_dpo_data.py extract \
#     --input "$source_file" --sft-data "$sft_file" --tokenizer "$base_model" \
#     --test "$test_file" --output "$input_file" --allow-missing-responses --seed "$seed"

# 2. Generate student rewrites under both instruction conditions.
"$python_bin" -u scripts/src/construct_dpo_data.py generate \
    --input "$input_file" --output "${prefix}_generated.jsonl" \
    --model "$base_model" --sft-adapter "$student_adapter" --model-name "$student_name" \
    --instruction-mode both --conversation-limit "$conversation_limit" \
    --max-model-len 4096 --max-new-tokens 512 --num-gpus "$num_gpus" \
    --tensor-parallel-size 1 --seed "$seed" --limit NULL

# 3. Score student answers.
"$python_bin" -u scripts/src/construct_dpo_data.py score \
    --input "${prefix}_generated.jsonl" --output "${prefix}_scored.jsonl" \
    --judge-model "$judge_model" --thinking-level low --batch-size 100 --workers 20 --limit NULL

# 4. Generate teacher answers and compare them against the selected student answers.
"$python_bin" -u scripts/src/construct_dpo_data.py compare \
    --input "${prefix}_scored.jsonl" --output "${prefix}_compared.jsonl" \
    --judge-model "$judge_model" --teacher-model "$teacher_model" \
    --min-applicability-score "$min_applicability" --max-rejected-score "$max_rejected_score" \
    --max-without-instruction-rejected-score "$max_noinst_rejected_score" \
    --thinking-level low --batch-size 100 --workers 20 --seed "$seed" --limit NULL

# 5. Keep teacher wins and apply response-length and full-sequence limits.
"$prepare_python" -u scripts/src/construct_dpo_data.py prepare \
    --input "${prefix}_compared.jsonl" --output "$output_file" \
    --test "$test_file" --tokenizer "$base_model" --max-seq-length 4096 \
    --min-applicability-score "$min_applicability" --max-rejected-score "$max_rejected_score" \
    --max-without-instruction-rejected-score "$max_noinst_rejected_score" \
    --max-with-instruction-length-ratio "$with_inst_length_ratio" \
    --max-without-instruction-length-ratio "$no_inst_length_ratio" --limit NULL
