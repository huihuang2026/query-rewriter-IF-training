#!/usr/bin/env bash
# Run in the foreground, e.g. inside tmux.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -f "$HOME/api-key.txt" ]]; then
    set -a
    source "$HOME/api-key.txt"
    set +a
fi

python_bin="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
input_file="data/query_rewriter_combined_selfins_compared_rejudge.json"
output_file="data/query_rewriter_combined_selfins_compared_rejudge_requality.json"
judge_model="gemini-4-argon"
batch_size=200
workers=50
seed=42
limit=NULL # Set to 100 for a trial; completed pairs count toward this limit.

# Check the latest Gemini wins with QUALITY_COMPARISON_PROMPT.
# Keep the existing with-instruction prompts and both responses unchanged.
# No additional score/length filter, response generation, or training.
# Resume quality judgments in a separate output; report verdict counts and batch ETA.
"$python_bin" -u scripts/src/rejudge_dpo_data_combined.py requality \
    --input "$input_file" --output "$output_file" \
    --judge-model "$judge_model" --batch-size "$batch_size" --workers "$workers" \
    --thinking-level low --max-output-tokens 2048 --seed "$seed" --limit "$limit" "$@"

# For later DPO preparation, use rejudge_dpo_data_combined.py prepare --requality
# so it validates the quality rubric and query_quality verdicts for these pairs.
