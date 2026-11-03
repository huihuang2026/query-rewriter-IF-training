#!/usr/bin/env bash
# Generate responses, publish completed scores, select SFT pairs, and optionally expand instructions.
# Construction shares one resumable output file; scoring uses a separate file. No training is launched.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == --help ]]; then
    echo "Usage: bash scripts/construct_sft_data_exported.sh [construct_sft_data.py options]"
    echo "Uses cleaned exported data; excludes customer training/test conversations and repeated export conversations."
    echo "Samples 400,000 original conversations using balanced quotas by customer_identifier."
    echo "Runs instruction generation, response generation, scoring, and score-based SFT sampling; expansion is optional."
    echo "Edit paths/settings below; comment out deduplication, instruction generation, or response generation separately."
    echo "Construction uses output_file; GPT scores are saved separately with a _scored suffix."
    echo "train_sft_exported.sh consumes the selected _sft.json file directly."
    exit 0
fi

if [[ -f "$HOME/api-key.txt" ]]; then
    source "$HOME/api-key.txt"
fi

python_bin="$HOME/miniconda3/bin/python"
input_file="data/query_rewriter_exported_cleaned.json"
seed_file="data/query_rewriter_exported_customer_max10_processed.json"
test_file="data/query_rewriter_exported_sample10k_random1k_filtered.json"
deduplicated_file="data/query_rewriter_exported_sample400k.json"
output_file="data/query_rewriter_exported_selfins.json"
scored_file="data/query_rewriter_exported_selfins_scored.json"
sft_file="data/query_rewriter_exported_selfins_sft.json"
sft_sample_size="50000" # Conversation pairs, each producing two SFT examples.
min_query_quality=4
min_instruction_following=3
scoring_model="gpt-6-luna"
instruction_model="gemini-3.5-flash-lite" # Changing this affects only missing instructions/summaries.
response_model="gemini-3.8-flash" # Keep fixed when resuming the same response output file.
batch_size=500
workers=30
seed_examples=5
variants_per_sample=1
sample_size=400000
initial_instruction_limit="100000"
response_limit="100000"
instruction_limit="400000"
sampling_seed=42

# # Step 1: Deduplicate normalized conversation text against customer training/test
# # conversations and within the exported data, then sample 400K conversations by company.
# # Small companies contribute all available records; larger companies receive balanced quotas.
# "$python_bin" -u scripts/src/construct_sft_data.py dedup \
#     --input "$input_file" --seeds "$seed_file" --test "$test_file" \
#     --output-dir "$(dirname -- "$deduplicated_file")" --output "$deduplicated_file" \
#     --sample-size "$sample_size" --seed "$sampling_seed"

# # Step 2: Summarize the first 100K conversations and generate user instructions only.
# # Existing instructions and answers in output_file are preserved.
# echo "Stage 1/5: construct-inst for $initial_instruction_limit conversations."
# "$python_bin" -u scripts/src/construct_sft_data.py construct-inst \
#     --input "$deduplicated_file" --seeds "$seed_file" --test "$test_file" \
#     --output-dir "$(dirname -- "$output_file")" --output "$output_file" \
#     --model "$instruction_model" --limit "$initial_instruction_limit" --seed "$sampling_seed" \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" \
#     --seed-examples "$seed_examples" --variants-per-sample "$variants_per_sample" "$@"

# # Step 3: Add both rewrites to existing instructions in the same output_file.
# # To evaluate/filter instructions first, comment out this block. response_limit counts
# # conversations with saved instructions. Missing instructions are skipped; if none exist, exit.
# echo "Stage 2/5: construct-resp for $response_limit conversations with saved instructions."
# "$python_bin" -u scripts/src/construct_sft_data.py construct-resp \
#     --input "$output_file" --seeds "$seed_file" --test "$test_file" \
#     --output-dir "$(dirname -- "$output_file")" --output "$output_file" \
#     --model "$response_model" --limit "$response_limit" --seed "$sampling_seed" \
#     --max-incomplete-pairs 10 \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" "$@"

# # Step 4: Score complete response pairs with the existing quality/following rubrics.
# # OPENAI_API_KEY is configured above. Scores resume separately.
# echo "Stage 3/5: score up to $response_limit pairs with $scoring_model into $scored_file."
# "$python_bin" -u scripts/src/construct_sft_data.py score \
#     --input "$output_file" --scored-output "$scored_file" \
#     --output-dir "$(dirname -- "$scored_file")" \
#     --scoring-model "$scoring_model" --limit "$response_limit" --seed "$sampling_seed" \
#     --thinking-level low --batch-size "$batch_size" --workers "$workers" "$@"

# # Step 5: Filter by BOTH scores, then randomly sample completed pairs for SFT.
# # Missing scores/responses, failed records, and test overlap are excluded.
# # A shortage stops sampling instead of lowering cutoffs or duplicating records.
# echo "Stage 4/5: sample $sft_sample_size pairs with quality >= $min_query_quality and instruction following >= $min_instruction_following."
# "$python_bin" -u scripts/src/prepare_sft_data.py \
#     --input "$scored_file" --output "$sft_file" --test "$test_file" \
#     --sample-pairs --instruction-count 1 --examples "$sft_sample_size" --seed "$sampling_seed" \
#     --min-query-quality "$min_query_quality" --min-instruction-following "$min_instruction_following"

# Step 6: Expand to 400K instructions total, reusing the first 100K instructions
# and preserving their responses. No responses are generated for the new instructions.
echo "Stage 5/5: construct-inst for $instruction_limit conversations total (resume existing instructions)."
"$python_bin" -u scripts/src/construct_sft_data.py construct-inst \
    --input "$deduplicated_file" --seeds "$seed_file" --test "$test_file" \
    --output-dir "$(dirname -- "$output_file")" --output "$output_file" \
    --model "$instruction_model" --limit "$instruction_limit" --seed "$sampling_seed" \
    --thinking-level low --batch-size "$batch_size" --workers "$workers" \
    --seed-examples "$seed_examples" --variants-per-sample "$variants_per_sample" "$@"
