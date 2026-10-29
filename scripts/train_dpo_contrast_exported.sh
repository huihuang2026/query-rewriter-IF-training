#!/usr/bin/env bash
# Prepare exported with-instruction contrast pairs, train from combined 200K SFT, then evaluate.
# Run after compare_dpo_data_contrast_exported.sh finishes; comment individual steps as needed.
# Comment individual steps to run them separately. Export OPENAI_API_KEY before evaluation.
set -euo pipefail
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="4,5,6,7"
python_bin="$PWD/.venv-finetune/bin/python"
prepare_python="$HOME/miniconda3/bin/python"
min_applicability_score=5
# Source gate on the original instructed response; not a score for the new negative.
max_source_following_score=1
with_instruction_length_ratio=2.0
suffix="${min_applicability_score}_${max_source_following_score}_${with_instruction_length_ratio}"
prefix="data/query_rewriter_exported_selfins_contrast"
compared_file="${prefix}_compared.json"
input_file="${prefix}_${suffix}_dpo_train.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
limit="NULL"

base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
sft_model_name="${base_model}-combined_selfins-200K_LoRA"
sft_adapter="ckpts/${sft_model_name}/final"
model_name="${sft_model_name}-exported_contrast_${suffix}-DPO"
checkpoint_dir="ckpts/${model_name}"
num_gpus=4
batch_size=1
gradient_accumulation=4
epochs=1
learning_rate="5e-6"
beta="0.1"
max_seq_length=4096
seed=42

# Step 1: Only keep judged Gemini wins; both answers share the instructed DPO prompt.
# Exclude test overlap and apply the chosen/rejected token-length ratio.
"$prepare_python" -u scripts/src/compare_dpo_data_contrast.py prepare \
    --input "$compared_file" --output "$input_file" --test "$test_file" \
    --min-applicability-score "$min_applicability_score" \
    --max-source-following-score "$max_source_following_score" \
    --tokenizer "$model" --max-with-instruction-length-ratio "$with_instruction_length_ratio" \
    --limit "$limit"

# Step 2: Train. Extra arguments apply only to training.
echo "Training $model_name on $num_gpus GPUs; effective batch: $((num_gpus * batch_size * gradient_accumulation)) preference pairs"
"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_dpo.py \
    --input "$input_file" --sft-adapter "$sft_adapter" \
    --output-dir "$checkpoint_dir" --epochs "$epochs" \
    --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
    --learning-rate "$learning_rate" --beta "$beta" \
    --max-seq-length "$max_seq_length" --seed "$seed" "$@"

# Step 3: Evaluate general quality without instructions and IF with instructions.
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
    --lora-path "$checkpoint_dir/final" --limit "$test_limit" --both-instruction-conditions
"$test_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions
"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
echo "Finished: $checkpoint_dir/final; evaluation: ${output_prefix}_evaluated.json"
