#!/usr/bin/env bash
# Select one training block by commenting/uncommenting it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES=0
python_bin="$PWD/.venv-finetune/bin/python"
num_gpus=1
epochs=1
batch_size=1
gradient_accumulation=8
learning_rate=5e-6
beta=0.1
max_seq_length=4096
seed=42

# 1. Ordinary DPO: continue the SFT adapter using the retained DPO data.
input_file="data/query_rewriter_self_instruct_1_4_with_noinst_length_dpo_train.json"
# input_file="data/query_rewriter_dpo_train.jsonl" # Newly constructed alternative.
initial_adapter="ckpts/Qwen3-4B-Instruct-2507-SFT/final"
model_name="Qwen3-4B-Instruct-2507-DPO"
sft_loss_weight=0.0

"$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
    scripts/src/train_dpo.py --input "$input_file" --sft-adapter "$initial_adapter" \
    --output-dir "ckpts/$model_name" --epochs "$epochs" --batch-size "$batch_size" \
    --gradient-accumulation "$gradient_accumulation" --learning-rate "$learning_rate" \
    --beta "$beta" --sft-loss-weight "$sft_loss_weight" --max-seq-length "$max_seq_length" \
    --seed "$seed" --strict-pairs --report-to none --run-name "$model_name"

# 2. Corrected-positive DPO: comment out block 1 and uncomment this block.
# input_file="data/query_rewriter_6kSFT_withinst_source20K_4_4_noinst_source20K_4_4_len1.0_dpo_train.json"
# # input_file="data/query_rewriter_dpo_contrast_train.jsonl" # Newly constructed alternative.
# initial_adapter="ckpts/Qwen3-4B-Instruct-2507-DPO/final"
# model_name="Qwen3-4B-Instruct-2507-DPO-contrast"
# sft_loss_weight=1.0
#
# "$python_bin" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$num_gpus" \
#     scripts/src/train_dpo.py --input "$input_file" --sft-adapter "$initial_adapter" \
#     --output-dir "ckpts/$model_name" --epochs "$epochs" --batch-size "$batch_size" \
#     --gradient-accumulation "$gradient_accumulation" --learning-rate "$learning_rate" \
#     --beta "$beta" --sft-loss-weight "$sft_loss_weight" --max-seq-length "$max_seq_length" \
#     --seed "$seed" --strict-pairs --report-to none --run-name "$model_name"
