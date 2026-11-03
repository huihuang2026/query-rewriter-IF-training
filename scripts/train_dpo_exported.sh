#!/usr/bin/env bash
# Prepare exported DPO pairs from cached comparisons, then train from combined 200K SFT and evaluate both instruction conditions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/train_dpo_exported.sh [train_dpo.py options]"
    echo "Edit the three score thresholds and two length ratios below for each experiment."
    echo "Preparation runs before training every time and preserves the comparison cache."
    echo "Data/checkpoint names include the source and all five thresholds."
    echo "Wider score ranges need completed comparisons: rerun construct_dpo_data_exported.sh compare first."
    echo "Evaluation automatically generates both instruction conditions, scores them, and prints metrics after training."
    echo "For evaluation only, comment out Steps 1 and 2. Extra arguments apply only to training."
    exit 0
fi

export CUDA_VISIBLE_DEVICES="4,5,6,7"
python_bin="$PWD/.venv-finetune/bin/python"
prepare_python="$HOME/miniconda3/bin/python"
data_dir="data"

# Five selection thresholds, carried over from construct_dpo_data_exported.sh.
min_applicability_score=1
max_rejected_score=4 # Instruction following with user instructions.
max_without_instruction_rejected_score=4 # Query quality without user instructions.
with_instruction_length_ratio=2.0
without_instruction_length_ratio=0.0
# Order: applicability, rejected with instruction, rejected without instruction, then the two ratios.
score_suffix="${min_applicability_score}_${max_rejected_score}_${max_without_instruction_rejected_score}"
ratio_suffix="${with_instruction_length_ratio}_${without_instruction_length_ratio}"
prefix="$data_dir/query_rewriter_exported_selfins"
compared_file="${prefix}_compared.json"
input_file="${prefix}_${score_suffix}_${ratio_suffix}_dpo_train.json"
test_file="$data_dir/query_rewriter_exported_sample10k_random1k_filtered.json"
limit="NULL"

base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
sft_model_name="${base_model}-combined_selfins-200K_LoRA"
sft_adapter="ckpts/${sft_model_name}/final"
model_name="${sft_model_name}-exported_${score_suffix}_${ratio_suffix}-DPO"
checkpoint_dir="ckpts/${model_name}"
num_gpus=4
batch_size=1
gradient_accumulation=4
epochs=1
learning_rate="5e-6"
beta="0.1"
max_seq_length=4096
seed=42

# # Step 1: Reapply the current thresholds to cached comparisons on every run.
# # Keep valid Gemini wins, exclude test overlap, and apply both token-length ratios.
# # Preparation reports kept counts for both instruction groups and leaves the cache untouched.
# echo "Preparing exported DPO data with thresholds ${score_suffix}_${ratio_suffix}: $input_file"
# "$prepare_python" -u scripts/src/construct_dpo_data.py prepare \
#     --input "$compared_file" --output "$input_file" --test "$test_file" \
#     --min-applicability-score "$min_applicability_score" --max-rejected-score "$max_rejected_score" \
#     --max-without-instruction-rejected-score "$max_without_instruction_rejected_score" \
#     --tokenizer "$model" \
#     --max-with-instruction-length-ratio "$with_instruction_length_ratio" \
#     --max-without-instruction-length-ratio "$without_instruction_length_ratio" --limit "$limit"

# # Step 2: Train. Extra arguments such as --resume-from-checkpoint apply only here.
# echo "Training $model_name on $num_gpus GPUs; effective batch: $((num_gpus * batch_size * gradient_accumulation)) preference pairs"
# "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
#     scripts/src/train_dpo.py \
#     --input "$input_file" --sft-adapter "$sft_adapter" \
#     --output-dir "$checkpoint_dir" --epochs "$epochs" \
#     --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
#     --learning-rate "$learning_rate" --beta "$beta" \
#     --max-seq-length "$max_seq_length" --seed "$seed" "$@"

# Step 3: Evaluate the trained DPO adapter after all training workers exit.
# For evaluation only, comment out Steps 1 and 2, keeping the model/path settings above.
if [[ -f "$HOME/api-key.txt" ]]; then
    source "$HOME/api-key.txt"
fi
test_python="$HOME/miniconda3/bin/python"
judge_model="gpt-6-astra"
test_limit="NULL"
# Use the first physical GPU assigned to this script: previous=0, exported=4.
test_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}"
mkdir -p outputs

# # Generate with and without user instructions in one inference run.
# CUDA_VISIBLE_DEVICES="$test_gpu" "$test_python" scripts/src/test_query_rewriter.py generate \
#     --input "$test_file" --output "${output_prefix}_generated.json" \
#     --model-path "$model" --model-name "$model_name" \
#     --lora-path "$checkpoint_dir/final" --limit "$test_limit" \
#     --both-instruction-conditions

# # Score general quality without instructions and instruction following with instructions.
# "$test_python" scripts/src/test_query_rewriter.py score \
#     --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
#     --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
#     --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
echo "Finished: $checkpoint_dir/final; evaluation: ${output_prefix}_evaluated.json"
