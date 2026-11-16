#!/usr/bin/env bash
# Edit the settings below; comment/uncomment the command you want to run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=0
python_bin="$PWD/.venv-finetune/bin/python"
base_model="models/Qwen3-4B-Instruct-2507"
input_file="data/query_rewriter_self_instruct_summary_processed.json"
# Use this instead after reconstructing SFT data:
# input_file="data/query_rewriter_sft_tokenized"
model_name="Qwen3-4B-Instruct-2507-SFT"
output_dir="ckpts/$model_name"
num_gpus=1
epochs=3
batch_size=2
gradient_accumulation=8
max_seq_length=4096
seed=42

# Train SFT. Add --sample-size 5000 to train on a subset.
# To resume, add --resume-from-checkpoint ckpts/<model-name>/checkpoint-<step>.
"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_sft.py --input "$input_file" --model "$base_model" --output-dir "$output_dir" \
    --mode lora --epochs "$epochs" --batch-size "$batch_size" \
    --gradient-accumulation "$gradient_accumulation" --max-seq-length "$max_seq_length" \
    --seed "$seed" --report-to none --run-name "$model_name"
