#!/usr/bin/env bash
# Prepare quality-checked instruction-following pairs, train from combined 200K SFT, and evaluate both instruction conditions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/train_dpo_combined_requality.sh [train_dpo.py options]"
    echo "Edit the two length ratios below for each experiment."
    echo "Preparation runs before training every time and preserves the comparison cache."
    echo "Data/checkpoint names include the source and both length ratios."
    echo "Evaluation automatically generates both instruction conditions, scores them, and prints metrics after training."
    echo "For evaluation only, comment out Steps 1 and 2. Extra arguments apply only to training."
    exit 0
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3"
python_bin="$PWD/.venv-finetune/bin/python"
prepare_python="$HOME/miniconda3/bin/python"
data_dir="data"

# Keep Gemini wins from the additional quality check; no new score thresholds.
# This source contains with-instruction pairs only.
with_instruction_length_ratio=2.0
without_instruction_length_ratio=0.0
# Order: with-instruction ratio, without-instruction ratio.
ratio_suffix="${with_instruction_length_ratio}_${without_instruction_length_ratio}"
prefix="$data_dir/query_rewriter_combined_selfins"
compared_file="${prefix}_compared_rejudge_requality.json"
input_file="${prefix}_rejudge_requality_${ratio_suffix}_dpo_train.json"
test_file="$data_dir/query_rewriter_exported_sample10k_random1k_filtered.json"
limit="NULL"

base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
sft_model_name="${base_model}-combined_selfins-200K_LoRA"
sft_adapter="ckpts/${sft_model_name}/final"
# Loss = DPO + this weight * chosen-answer SFT cross-entropy.
sft_loss_weight=0.0
epochs=3 # Total epochs, including the completed epoch in the resumed checkpoint.
previous_model_name="${sft_model_name}-combined_rejudge_requality_${ratio_suffix}-DPO"
resume_checkpoint="ckpts/${previous_model_name}/checkpoint-936"
model_name="${previous_model_name}-epochs${epochs}"
checkpoint_dir="ckpts/${model_name}"
num_gpus=4
batch_size=1
gradient_accumulation=4
learning_rate="5e-6"
beta="0.1"
max_seq_length=4096
seed=42

# W&B login: .venv-finetune/bin/wandb login (once), or WANDB_API_KEY in ~/api-key.txt.
export WANDB_ENTITY="damian796-personal"
export WANDB_PROJECT="query-rewriter-IF-training"
export WANDB_LOG_MODEL="false"
if [[ -f "$HOME/api-key.txt" ]]; then
    set -a
    source "$HOME/api-key.txt"
    set +a
fi

# # --requality validates the quality rubric and query_quality verdicts on instructed pairs.
# # Step 1: Select new Gemini wins and apply the two length ratios.
# # Keep valid Gemini wins, exclude test overlap, and apply both token-length ratios.
# # Preparation reports kept counts for both instruction groups and leaves the cache untouched.
# echo "Preparing combined requality DPO data with length ratios ${ratio_suffix}: $input_file"
# "$prepare_python" -u scripts/src/rejudge_dpo_data_combined.py prepare --requality \
#     --input "$compared_file" --output "$input_file" --test "$test_file" \
#     --tokenizer "$model" \
#     --max-with-instruction-length-ratio "$with_instruction_length_ratio" \
#     --max-without-instruction-length-ratio "$without_instruction_length_ratio" --limit "$limit"

# Step 2: Train. Extra arguments such as --resume-from-checkpoint apply only here.
echo "Training $model_name on $num_gpus GPUs; effective batch: $((num_gpus * batch_size * gradient_accumulation)) preference pairs"
"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_dpo.py \
    --input "$input_file" --sft-adapter "$sft_adapter" \
    --output-dir "$checkpoint_dir" --epochs "$epochs" \
    --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
    --learning-rate "$learning_rate" --beta "$beta" --sft-loss-weight "$sft_loss_weight" \
    --max-seq-length "$max_seq_length" --seed "$seed" \
    --resume-from-checkpoint "$resume_checkpoint" \
    --report-to wandb --run-name "$model_name" "$@"

# Step 3: Evaluate the trained DPO adapter after all training workers exit.
# For evaluation only, comment out Steps 1 and 2, keeping the model/path settings above.
test_python="$HOME/miniconda3/bin/python"
judge_model="gpt-6-astra"
test_limit="NULL"
# Evaluate on the first physical GPU assigned to this script.
test_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}"
mkdir -p outputs

# Generate with and without user instructions in one inference run.
CUDA_VISIBLE_DEVICES="$test_gpu" "$test_python" scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$model" --model-name "$model_name" \
    --lora-path "$checkpoint_dir/final" --limit "$test_limit" \
    --both-instruction-conditions

# Score general quality without instructions and instruction following with instructions.
"$test_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
echo "Finished: $checkpoint_dir/final; evaluation: ${output_prefix}_evaluated.json"
