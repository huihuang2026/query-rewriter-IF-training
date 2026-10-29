#!/usr/bin/env bash
# Reuse step-3 scores/answers; compare instructed Gemini against uninstructed SFT.
# Export GEMINI_API_KEY before running. Run in the foreground (e.g. inside tmux).
set -euo pipefail
export GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

python_bin="$HOME/miniconda3/bin/python"
prefix="data/query_rewriter_previous_selfins"
scored_file="${prefix}_unused_scored.json"
candidates_file="${prefix}_contrast_candidates.json"
compared_file="${prefix}_contrast_compared.json"
compare_model="gemini-4-argon"
teacher_model="gemini-3.8-flash"
min_applicability_score=5
# This is the ORIGINAL instructed SFT answer's IF score, used for source selection.
# The uninstructed answer has no saved IF score under the instructed prompt.
max_source_following_score=1
batch_size=500
workers=50
seed=42
limit="NULL"

# Join exact matching conversations/model configurations, then compare under the
# separately labeled base/user instructions, emphasizing the user instruction.
# Missing instructed teacher answers are generated as before.
# Results have separate contrast IDs/files; ordinary comparisons are never reused.
"$python_bin" -u scripts/src/compare_dpo_data_contrast.py compare \
    --input "$scored_file" --candidates-output "$candidates_file" --output "$compared_file" \
    --judge-model "$compare_model" --teacher-model "$teacher_model" \
    --min-applicability-score "$min_applicability_score" \
    --max-source-following-score "$max_source_following_score" \
    --batch-size "$batch_size" --workers "$workers" --seed "$seed" --limit "$limit"
