#!/usr/bin/env bash
# Train combined 200K SFT on the archived DPO dataset, then evaluate both instruction conditions.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/train_dpo_old.sh [train_dpo.py options]"
    echo "Uses the existing 52,186 pairs directly: 34,791 with instruction + 17,395 without."
    echo "Starts from combined 200K SFT; the saved rejected answers came from the older SFT model."
    echo "For evaluation only, comment out Step 1. Extra arguments apply only to training."
    exit 0
fi

export CUDA_VISIBLE_DEVICES="4,5,6,7"
python_bin="$PWD/.venv-finetune/bin/python"
input_file="data/trash-bin/export-data-dpo/query_rewriter_self_instruct_1_4_with_noinst_length_half_dpo_train.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"

base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
sft_model_name="${base_model}-combined_selfins-200K_LoRA"
sft_adapter="ckpts/${sft_model_name}/final"
model_name="${sft_model_name}-old_1_4_with_noinst_length_half-DPO"
checkpoint_dir="ckpts/${model_name}"
num_gpus=4
batch_size=1
gradient_accumulation=4
epochs=1
learning_rate="5e-6"
beta="0.1"
max_seq_length=4096
seed=42

# Step 1: Train directly on the existing prompt/chosen/rejected pairs.
# No resampling, comparison, or score/length filtering is performed here.
# The Python trainer still excludes identical and overlength pairs.
echo "Training $model_name on $num_gpus GPUs; effective batch: $((num_gpus * batch_size * gradient_accumulation)) preference pairs"
"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_dpo.py \
    --input "$input_file" --sft-adapter "$sft_adapter" \
    --output-dir "$checkpoint_dir" --epochs "$epochs" \
    --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
    --learning-rate "$learning_rate" --beta "$beta" \
    --max-seq-length "$max_seq_length" --seed "$seed" "$@"

# Step 2: Evaluate the trained adapter after all training workers exit.
# Evaluation credentials are configured here, as in train_dpo_exported.sh.
if [[ -f "$HOME/api-key.txt" ]]; then
    source "$HOME/api-key.txt"
fi
test_python="$HOME/miniconda3/bin/python"
judge_model="gpt-6-astra"
test_limit="NULL"
test_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}"
mkdir -p outputs

CUDA_VISIBLE_DEVICES="$test_gpu" "$test_python" scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$model" --model-name "$model_name" \
    --lora-path "$checkpoint_dir/final" --limit "$test_limit" \
    --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
echo "Finished: $checkpoint_dir/final; evaluation: ${output_prefix}_evaluated.json"
