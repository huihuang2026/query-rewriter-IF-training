#!/usr/bin/env bash
# Controlled negative-generator experiment. Run in tmux; comment stages independently.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python_bin="$PWD/.venv-finetune/bin/python"
prepare_python="$HOME/miniconda3/bin/python"
model="$PWD/models/Qwen3-4B-Instruct-2507"
sft_model_name="Qwen3-4B-Instruct-2507-combined_selfins-200K_LoRA"
sft_adapter="$PWD/ckpts/$sft_model_name/final"
old_model_name="Qwen3-4B-Instruct-2507-self_instruct_summary_1_all-LoRA-checkpoint-72426"
old_adapter="$PWD/ckpts/trash-bin/Qwen3-4B-Instruct-2507-self_instruct_summary_1_all-LoRA/checkpoint-72426"
source_file="data/query_rewriter_previous_selfins_unused_scored.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
sample_size=10000
min_applicability_score=2
seed=42
max_seq_length=4096
max_new_tokens=512
prefix="data/compare/query_rewriter_previous_negative_source_app${min_applicability_score}_${sample_size}_seed${seed}"
shared_sample="${prefix}_shared.json"
# Separate caches and training artifacts from the earlier DPO-generator experiment.
old_generated="${prefix}_old_sft72426_generated.json"
input_file="${prefix}_old_sft72426_dpo_train.json"
model_name="${sft_model_name}-negative_source_old_sft72426_app${min_applicability_score}_${sample_size}_seed${seed}-DPO"
checkpoint_dir="ckpts/$model_name"
num_gpus=4
batch_size=1
gradient_accumulation=4
epochs=1
learning_rate=5e-6
beta=0.1

if [[ -f "$HOME/api-key.txt" ]]; then
    set -a
    source "$HOME/api-key.txt"
    set +a
fi
export WANDB_ENTITY="damian796-personal"
export WANDB_PROJECT="query-rewriter-IF-training"
export WANDB_LOG_MODEL=false

# 1. Freeze the SAME 10K prompts and Gemini positives for both experiments.
# Applicability >= 2 only; no quality/IF score, winner, or length-ratio selection.
# Exclude failed/missing answers, held-out overlap, and context overflow.
"$prepare_python" -u scripts/src/prepare_negative_model_comparison.py sample \
    --input "$source_file" --output "$shared_sample" --test "$test_file" \
    --size "$sample_size" --applicability "$min_applicability_score" --seed "$seed" \
    --model "$model" --negative-model "$sft_model_name" \
    --max-length "$max_seq_length" --max-new-tokens "$max_new_tokens"

# 2a. Regenerate ONLY negatives with the older adapter, using the full instructed prompts.
# Match cached decoding: temperature=0, max_new_tokens=512, no thinking.
"$prepare_python" -u scripts/src/construct_dpo_data.py generate \
    --input "$shared_sample" --output "$old_generated" \
    --model "$model" --sft-adapter "$old_adapter" --model-name "$old_model_name" \
    --num-gpus "$num_gpus" --max-model-len "$max_seq_length" --max-new-tokens "$max_new_tokens"

# 2b. Align by ID and verify prompts/positives are unchanged; replace only rejected answers.
"$prepare_python" -u scripts/src/prepare_negative_model_comparison.py pairs \
    --sample "$shared_sample" --generated "$old_generated" --output "$input_file" \
    --negative-model "$old_model_name" --model "$model" --max-length "$max_seq_length"

# 3. Both runs start from the SAME current SFT adapter, NOT the old generator.
# Strict pairs prevents model-specific dropping; identical pairs remain as zero-signal examples.
"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_dpo.py \
    --input "$input_file" --sft-adapter "$sft_adapter" --output-dir "$checkpoint_dir" \
    --epochs "$epochs" --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
    --learning-rate "$learning_rate" --beta "$beta" --sft-loss-weight 0 \
    --max-seq-length "$max_seq_length" --seed "$seed" --strict-pairs \
    --report-to wandb --run-name "$model_name" "$@"

# 4. Same test set and judge in both runs. Comment steps 1-3 for evaluation only.
test_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}"
mkdir -p outputs
CUDA_VISIBLE_DEVICES="$test_gpu" "$prepare_python" scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$model" --model-name "$model_name" --lora-path "$checkpoint_dir/final" \
    --limit NULL --both-instruction-conditions
"$prepare_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model gpt-6-astra --limit NULL --batch-size 100 --workers 20 --both-instruction-conditions
"$prepare_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
echo "Finished: $checkpoint_dir/final"
