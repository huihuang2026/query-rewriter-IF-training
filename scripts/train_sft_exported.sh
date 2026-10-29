#!/usr/bin/env bash
# Prepare the score-selected exported SFT sample, then train LoRA; testing is optional.
export CUDA_VISIBLE_DEVICES=4,5,6,7
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/train_sft_exported.sh [train_sft.py options]"
    echo "Reads query_rewriter_exported_selfins_sft.json selected by construct_sft_data_exported.sh, then prepares and trains."
    echo "No new sampling: the construction script owns score filtering and SFT sample size."
    echo "Both responses must exist and be nonempty; instruction-only, failed, and held-out records are excluded."
    echo "Edit input_file directly in this script."
    echo "Edit data/model paths and training parameters directly in this script."
    echo "Testing is commented out; uncomment its block to generate, score, and calculate metrics."
    echo "For evaluation only, comment out preparation and training, then uncomment testing."
    exit 0
fi

python_bin="$PWD/.venv-finetune/bin/python"
data_dir="data"
input_file="$data_dir/query_rewriter_exported_selfins_sft.json"
processed_file="$data_dir/query_rewriter_exported_selfins_processed.jsonl"
tokenized_dir="$data_dir/query_rewriter_exported_selfins_tokenized"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
instruction_count="1"
sampling_seed=42
base_model="Qwen3-4B-Instruct-2507"
model="$PWD/models/$base_model"
model_name="${base_model}-exported_selfins-100K_LoRA"
checkpoint_dir="ckpts/${model_name}"
epochs="3"
num_gpus="4"
batch_size="2"
gradient_accumulation="2"
max_seq_length="4096"

# The SFT sample is created after scoring in construct_sft_data_exported.sh.
if [[ ! -s "$input_file" ]]; then
    echo "Selected SFT sample missing: $input_file. Run the construction script's score-based sampling step first." >&2
    exit 1
fi

# Step 1: Convert the selected pairs to prompt/completion examples and tokenize.
# No second downsampling: each selected pair supplies its two actual responses.
# Source hashes and preparation settings control cache reuse.
rebuild_sft_data=0
prepare_options=()
if [[ "$rebuild_sft_data" == 1 ]]; then
    prepare_options=(--rebuild-cache)
fi
# "$python_bin" -u scripts/src/prepare_sft_data.py \
#     --input "$input_file" --output "$processed_file" --test "$test_file" \
#     --instruction-count "$instruction_count" --examples NULL --seed "$sampling_seed" \
#     --tokenized-output "$tokenized_dir" --model "$model" --max-seq-length "$max_seq_length" \
#     "${prepare_options[@]}"

# # Step 2: Train. Extra arguments apply only to this command.
# echo "Training $model_name on $num_gpus GPUs; effective batch size: $((num_gpus * batch_size * gradient_accumulation))"
# "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
#     scripts/src/train_sft.py \
#     --input "$tokenized_dir" --output-dir "$checkpoint_dir" \
#     --model "$model" --mode lora --epochs "$epochs" \
#     --batch-size "$batch_size" --gradient-accumulation "$gradient_accumulation" \
#     --max-seq-length "$max_seq_length" "$@"

# Step 3: Optional testing. Uncomment this block to evaluate after training.
# For testing only, comment out the preceding preparation/sampling and training blocks.
# Export OPENAI_API_KEY before scoring.
test_python="$HOME/miniconda3/bin/python"
judge_model="gpt-6-astra"
test_limit="NULL"
test_gpu=0
export OPENAI_API_KEY="${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}"
test_name="${test_file##*/}"
output_prefix="outputs/${test_name%.json}_${model_name}"
mkdir -p outputs
exec 9>"${output_prefix}.lock"
if ! flock -n 9; then
    echo "An evaluation for $model_name is already running." >&2
    exit 1
fi

# Generate with and without user instructions in one inference run.
CUDA_VISIBLE_DEVICES="$test_gpu" "$test_python" scripts/src/test_query_rewriter.py generate \
    --input "$test_file" --output "${output_prefix}_generated.json" \
    --model-path "$model" --model-name "$model_name" \
    --lora-path "$checkpoint_dir/final" --limit "$test_limit" \
    --both-instruction-conditions

# Score quality without user instructions and instruction following with them.
"$test_python" scripts/src/test_query_rewriter.py score \
    --input "${output_prefix}_generated.json" --output "${output_prefix}_evaluated.json" \
    --judge-model "$judge_model" --limit "$test_limit" --batch-size 100 --workers 20 \
    --both-instruction-conditions

"$test_python" scripts/src/test_query_rewriter.py calculate \
    --input "${output_prefix}_evaluated.json" | tee "${output_prefix}_metrics.txt"
