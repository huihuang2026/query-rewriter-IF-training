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
combined_file="data/query_rewriter_combined_selfins_compared.json"
rejudged_file="data/query_rewriter_combined_selfins_compared_rejudge_concise_length_filtered.json"
tokenizer="$PWD/models/Qwen3-4B-Instruct-2507"
judge_model="gemini-4-argon"
max_rejected_score=4 # Student query-quality score; no instruction applicability gate.
batch_size=200
workers=50
seed=42
limit=NULL # First N eligible original Gemini wins without instructions; NULL for all.

# Reuse the combined comparison cache and existing teacher/student answers.
# Only original Gemini wins without user instructions are rejudged.
# Skip Gemini answers longer than the student (tokens) before the limit/API calls.
# New judgments resume in a separate file; use a fresh output if changing the rubric.
"$python_bin" -u scripts/src/rejudge_dpo_data_combined.py rejudge \
    --input "$combined_file" --output "$rejudged_file" \
    --without-user-instruction --tokenizer "$tokenizer" --max-rejected-score "$max_rejected_score" \
    --judge-model "$judge_model" --batch-size "$batch_size" --workers "$workers" \
    --thinking-level low --max-output-tokens 2048 --seed "$seed" --limit "$limit" "$@"

# For later preparation, use this Python entry point's prepare command with
# --without-user-instruction so it validates the same quality rubric.
