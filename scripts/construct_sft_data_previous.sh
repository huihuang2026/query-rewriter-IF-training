#!/usr/bin/env bash
# Reproduce previous-data construction, Gemini scoring, and score-filtered SFT sampling.
# All stages remain commented because this dataset and its SFT sample already exist.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/construct_sft_data_previous.sh [construct_sft_data.py options]"
    echo "Edit input_file, seed_file, test_file, and output_file directly in this script."
    echo "Uncomment individual generation, scoring, or SFT sampling blocks as needed."
    echo "Existing instructions, matching responses, and scores are reused. Scoring uses gemini-3.5-flash-lite."
    echo "Generation shares output_file; judgments use _scored. The response stage requires saved instructions."
    echo "Sampling defaults reproduce the previous selection: 50,000 pairs, quality >= 5, instruction following >= 5."
    exit 0
fi

export GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}"
python_bin="$HOME/miniconda3/bin/python"
input_file="data/Query-Rewriter-Dataset/query_rewriter_cleaned.jsonl"
seed_file="data/query_rewriter_exported_customer_max10_processed.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
output_file="data/query_rewriter_previous_selfins.json"
scored_file="data/query_rewriter_previous_selfins_scored.json"
sft_file="data/query_rewriter_previous_selfins_sft.jsonl"
scoring_model="gemini-3.5-flash-lite" # Previous experiment used Gemini, not the exported GPT scorer.
scoring_limit="NULL" # Score the full existing pool before sampling.
sft_sample_size="50000" # Conversation pairs, each producing two SFT examples.
min_query_quality=5
min_instruction_following=5
instruction_model="gemini-3.8-flash" # Changing this affects only missing instructions/summaries.
response_model="gemini-3.8-flash" # Keep fixed when resuming the same response output file.
batch_size=500
workers=50
seed_examples=5
variants_per_sample=1 # Existing records with two variants remain preserved.
instruction_limit="50000"
response_limit="50000"
sampling_seed=42

# # Step 1: Summarize conversations and generate user instructions only (no rewrites).
# # Existing instructions and answers in output_file are preserved.
# "$python_bin" -u scripts/src/construct_sft_data.py construct-inst \
#     --input "$input_file" --seeds "$seed_file" --test "$test_file" \
#     --output-dir "$(dirname -- "$output_file")" --output "$output_file" \
#     --model "$instruction_model" --limit "$instruction_limit" --seed "$sampling_seed" \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" \
#     --seed-examples "$seed_examples" --variants-per-sample "$variants_per_sample" "$@"

# # Step 2: Add both rewrites to existing instructions in the same output_file.
# # To evaluate/filter instructions first, leave this block commented. response_limit counts
# # conversations with saved instructions, keeping their existing variants and matching responses.
# # Missing instructions are skipped; if none exist, exit without generating any instructions.
# "$python_bin" -u scripts/src/construct_sft_data.py construct-resp \
#     --input "$output_file" --seeds "$seed_file" --test "$test_file" \
#     --output-dir "$(dirname -- "$output_file")" --output "$output_file" \
#     --model "$response_model" --limit "$response_limit" --seed "$sampling_seed" \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" "$@"

# # Step 3: Score complete pairs using the previous experiment's Gemini scorer.
# # Score quality without user instructions and instruction following with them.
# # Existing matching judgments are reused; only fully scored rows are published.
# "$python_bin" -u scripts/src/construct_sft_data.py score \
#     --input "$output_file" --scored-output "$scored_file" \
#     --output-dir "$(dirname -- "$scored_file")" \
#     --scoring-model "$scoring_model" --limit "$scoring_limit" --seed "$sampling_seed" \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" "$@"

# # Step 4: Filter both scores, then randomly sample 50K complete conversation pairs.
# # Defaults match the historical 5/5 selection; adjust cutoffs here for a new experiment.
# # Failed/empty pairs, missing scores, and test overlap are excluded. A shortage exits.
# # Running this block replaces sft_file; leave commented to preserve the current sample.
# "$python_bin" -u scripts/src/prepare_sft_data.py \
#     --input "$scored_file" --output "$sft_file" --test "$test_file" \
#     --sample-pairs --instruction-count 1 --examples "$sft_sample_size" --seed "$sampling_seed" \
#     --min-query-quality "$min_query_quality" --min-instruction-following "$min_instruction_following"
