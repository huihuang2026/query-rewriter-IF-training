#!/usr/bin/env bash
# Run in the foreground, e.g. inside tmux. Comment stages independently if needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ -f "$HOME/api-key.txt" ]]; then
    set -a
    source "$HOME/api-key.txt"
    set +a
fi

python_bin="${PYTHON_BIN:-$HOME/miniconda3/bin/python}"
exported_file="data/query_rewriter_exported_selfins_compared.json"
previous_file="data/query_rewriter_previous_selfins_compared.json"
combined_file="data/query_rewriter_combined_selfins_compared.json"
rejudged_file="data/query_rewriter_combined_selfins_compared_rejudge.json"
judge_model="gemini-4-argon"
min_applicability_score=1
max_rejected_score=4
batch_size=200
workers=50
seed=42
limit=NULL # First N eligible original Gemini wins with instructions; NULL for all.

# # 1. Combine both groups and retain the latest row per ID, including old verdicts.
# "$python_bin" -u scripts/src/rejudge_dpo_data_combined.py combine \
#     --inputs "$exported_file" "$previous_file" --output "$combined_file"

# 2. Rejudge ONLY original Gemini-winning with-user-instruction pairs.
# Original verdicts do not count as completed work. Only this new output resumes;
# changed rubric/model/answers require a fresh output file rather than stale reuse.
# Missing/failed answers are reported and skipped, never generated here.
# Filter before judging: applicability >= minimum and student IF <= maximum.
# Original student wins/ties and without-instruction pairs are excluded before the limit/API calls.
# Earlier out-of-scope judgments are backed up and removed from the active output on rerun.
# Changing thresholds reuses completed judgments and keeps cached rows outside the range.
# Extra arguments such as --dry-run apply only to this step.
"$python_bin" -u scripts/src/rejudge_dpo_data_combined.py rejudge \
    --input "$combined_file" --output "$rejudged_file" \
    --judge-model "$judge_model" --batch-size "$batch_size" --workers "$workers" \
    --min-applicability-score "$min_applicability_score" --max-rejected-score "$max_rejected_score" \
    --thinking-level low --max-output-tokens 2048 --seed "$seed" --limit "$limit" "$@"

# Later use scripts/src/rejudge_dpo_data_combined.py prepare (same options as
# construct_dpo_data.py prepare) to validate the new rubric and select DPO pairs.
