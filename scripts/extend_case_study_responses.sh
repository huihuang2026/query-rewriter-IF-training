#!/usr/bin/env bash
# Add two responses per model to the existing 35-sample CSV. No judging or training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python_bin="$HOME/miniconda3/bin/python"
test_gpu=0
base_model="$PWD/models/Qwen3-4B-Instruct-2507"
input_csv="data/query_rewriter_sample_35_selected_model_responses - Sheet1.csv"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
output_csv="data/query_rewriter_sample_35_selected_model_responses_extended.csv"
output_dir="outputs/case_study_35_extended"
selected_file="$output_dir/selected_test_records.json"
max_new_tokens=256
mkdir -p "$output_dir"

# Add further checkpoint directory names here to append two more columns each.
models=(
    "Qwen3-4B-Instruct-2507-700K-SFT-34K-DPO-app5-rej2-17K-DPO-length-control"
    "Qwen3-4B-Instruct-2507-combined_selfins-200K_LoRA-old_1_4_with_noinst_length_half-DPO"
)

"$python_bin" scripts/src/extend_case_study_responses.py prepare \
    --input-csv "$input_csv" --test-file "$test_file" --output "$selected_file"

# One process per model releases GPU memory before the next model loads.
# Both instruction conditions are submitted together in each inference run.
model_outputs=()
for model_name in "${models[@]}"; do
    generated_file="$output_dir/${model_name}_generated.json"
    CUDA_VISIBLE_DEVICES="$test_gpu" "$python_bin" scripts/src/test_query_rewriter.py generate \
        --input "$selected_file" --output "$generated_file" \
        --model-path "$base_model" --model-name "$model_name" \
        --lora-path "$PWD/ckpts/$model_name/final" \
        --both-instruction-conditions --max-new-tokens "$max_new_tokens"
    model_outputs+=(--model-output "$model_name" "$generated_file")
done

"$python_bin" scripts/src/extend_case_study_responses.py export \
    --input-csv "$input_csv" --test-file "$test_file" --output "$output_csv" \
    "${model_outputs[@]}"
