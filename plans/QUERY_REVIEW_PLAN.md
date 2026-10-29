# Conversation assessment pipeline

The input is `data/export-data-processed/query_rewriter_exported_cleaned.json`. Despite the `.json` extension, it contains JSON Lines: one record per line. Records include `user_content`, `rewritten_query`, `system_prompt`, timestamps, and call identifiers. All source fields are preserved.

`scripts/process_exported.sh` skips export records that cannot be converted, including truncated nested JSON fields. Each exclusion is logged with its source file, line number, error, and available record ID in `data/export-data-processed/query_rewriter_exported_cleaned.errors.json`. Original export files are unchanged. The final processed file is replaced only after processing completes; the Python converter remains strict when called without `--skip-invalid`.

The workflow has three separate commands, followed by optional ranking:

1. `language`: mark each conversation's language locally. Keep every record; do not filter by language, length, or content.
2. `quality`: independently score conversation quality from **0 to 5**.
3. `retrieval`: independently decide **yes/no** whether the latest user request needs uploaded-KB retrieval.
4. `rank`: sort by conversation-quality score, highest first.

There is no rewritten-query quality assessment or three-score average anymore. Both API judges receive only the conversation. They do not receive the rewritten query, the source rewriter's system prompt, or the other judge's assessment. Each assessment returns only its score/decision and a brief English reason.

## Setup and run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate query-writer
python -m pip install pycld2==0.42 openai pydantic tqdm
bash scripts/construct_test_data.sh
```

The shell script defaults to the first **100 original source lines**, batches of **50**, and **10 worker threads**. Edit `limit`, `batch_size`, `workers`, and `model` at its top. Increasing `limit` allows a larger run; nothing starts a full-dataset API assessment automatically during setup. The shell sets `OPENAI_API_KEY` from the environment or privately prompts when it is absent. It contains no embedded API key.

The shell runs language marking, then keeps only `en` and `zh` records in `language_filtered`. Quality scoring runs on that file. Next it keeps only records whose quality score exactly equals `quality_score` (default `5`) in `quality_filtered`. Retrieval assessment runs only on that file. Change `quality_score` at the top of the shell script to choose a different exact score; this is not a minimum-score threshold. The filters preserve record contents and order, overwrite their outputs, and make no API calls.

## Language marking

`pycld2` runs locally and annotates only the conversation (`user_content`), after stripping speaker labels in a detection copy. The original text remains unchanged. `language.code` contains a language code such as `en`, `zh`, or `de`; `mixed` marks substantial detected multilingual content, and `unknown` marks uncertain/empty text. Reliability and detected language percentages are included; percentages are estimated text shares, not confidence probabilities.

Foreign languages, unreliable detection, short greetings, missing queries, and low-quality conversations are all retained. Malformed JSON lines are retained as `source.raw_line`; missing/non-text conversation content is marked unknown and cannot be API-assessed. Missing/non-text conversation rows and inputs above `--max-chars` are retained in assessment outputs with a skipped status, rather than silently removed. Empty string conversations can be assessed and should receive quality 0.

```bash
python scripts/src/construct_test_data.py language \
  --input data/export-data-processed/query_rewriter_exported_cleaned.json \
  --output-dir outputs --suffix exported_head100 --limit 100 --overwrite
```

## Conversation-quality assessment

The structured result is stored in `review.conversation_quality`, with `reason` first, then `score`. Example: `{"reason": "Clear and meaningful conversation.", "score": 4}`.

| Score | Meaning |
|---|---|
| 0 | Empty/nonsense/unintelligible, or purely greetings, jokes, or role-play with no substantive information need |
| 1 | Very little useful content; fragmented or incomplete intent is uninterpretable |
| 2 | Some topical content, but substantial ambiguity, weak context, or an unproductive exchange |
| 3 | Meaningful and useful intent is understandable, with notable context/response limitations |
| 4 | Coherent substantive intent and useful context, with only minor gaps or disfluencies |
| 5 | Especially clear, meaningful interaction with concrete needs, useful contributions, and sufficient context |

Quality is not determined by length or turn count. Short substantive questions and brief follow-ups can be useful. A greeting or joke within a meaningful conversation is not an automatic rejection. Retrieval-free conversations can still have high conversation quality. Agent claims are context, not independently verified facts.

```bash
python scripts/src/construct_test_data.py quality \
  --input outputs/language_marked_exported_head100.json \
  --output-dir outputs --suffix exported_head100 --limit 100 --overwrite \
  --batch-size 50 --workers 10
```

## Retrieval-necessity assessment

This command reads the quality output, or a subset you produce from it, and preserves its language marks, source fields, and quality assessment. It adds `review.retrieval_necessity`, with `reason` first, then `needed` (a boolean determination). Example: `{"reason": "The question requires the organization’s pricing policy.", "needed": true}`.

- `needed: true` means **yes**: the current request needs external facts plausibly in an uploaded KB, such as policies, prices, opening hours, facilities, technical details, or eligibility conditions, not already answered in the supplied context.
- `needed: false` means **no**: greetings/jokes, acknowledgments, contact collection, transfers, already-answered questions, ordinary reasoning, or a request solely for an action/live account lookup rather than KB facts.

This is a yes/no assessment of uploaded-KB retrieval necessity, separate from conversation quality.

```bash
python scripts/src/construct_test_data.py retrieval \
  --input outputs/conversation_quality_exported_head100.json \
  --output-dir outputs --suffix exported_head100 --limit 100 --overwrite \
  --batch-size 50 --workers 10
```

## Outputs and execution behavior

All outputs are files directly under `outputs`, using the selected suffix. Record files retain JSON Lines content with a `.json` extension.

| File prefix | Contents |
|---|---|
| `language_marked` | Every input record with `record_id`, `source_line`, original `source`, and `language` metadata |
| `language_summary` | Annotation counts by language; marking itself removes no records |
| `language_filtered` | Only records marked `en` or `zh` |
| `quality_filtered` | Only records with the exact selected conversation-quality score |
| `conversation_quality` | Language-marked records with the 0–5 quality result |
| `retrieval_necessity` | Quality records with the independent boolean retrieval result added |
| `quality_review`, `retrieval_review` | CSVs with model decisions and blank manual-review columns |
| `quality_config`, `retrieval_config` | Independent rubrics, schemas, models, and input fingerprints |
| `quality_summary`, `retrieval_summary` | Stage attempt/status counts, execution settings, and token usage |

`assessments.quality` and `assessments.retrieval` preserve each stage's success/error status and API metadata independently. The top-level `status` refers to the most recently run stage. The assessment commands themselves impose no filtering; the shell’s two explicit filter commands select which rows reach each stage.

Concurrent requests run within one batch at a time. Each batch has a progress bar with elapsed time, ETA, and throughput. Results are written in input order even if responses complete out of order. All results in an errored batch are saved, then that stage stops before starting another batch. The shell's `set -e` prevents continuing to retrieval if quality scoring fails.

The shell uses `--overwrite`, so each run performs fresh assessments and replaces its stage outputs and manual CSVs. To resume instead, omit `--overwrite` from the desired Python assessment command. Successful/skipped rows are reused and failed rows retried, without duplicate final records. Model/rubric/input changes require overwrite or a different suffix; batch size and worker count can change without invalidating existing scores. SDK retries can add HTTP requests and an interrupted unfinished batch may be repeated; reported usage cannot include unknown usage from failed requests.

## Manual review and ranking

Open the stage CSVs and fill `human_conversation_quality` (0–5), `human_retrieval_needed` (true/false), and optional `human_notes`. Inspect the decisions and reasons, revise the independent rubric if needed, and run a new iteration before scaling up. No automatic calibration decision or final test-set selection is made.

Ranking uses **only the 0–5 conversation-quality score**, since retrieval necessity is now a boolean. Ties retain input order; unscored quality results appear last. The ranked file contains only `record_id`, `source`, and `review`, as requested, while full stage outputs retain language and execution metadata.

```bash
python scripts/src/construct_test_data.py rank \
  --input outputs/retrieval_necessity_exported_head100.json \
  --output outputs/retrieval_necessity_exported_head100_ranked.json
```

## Full customer-identifier distribution

`scripts/customer_distribution.py` streams the entire processed export and counts every `customer_identifier`; it does not sample or change the input. It saves complete counts and percentages to `outputs/customer_distribution.json` and `.csv`. Charts include every customer: a compact overview, full labeled linear/logarithmic PNG and SVG charts, and a paginated PDF with all identifier labels. The logarithmic versions make smaller groups visible; the linear versions show absolute count differences.

```bash
.venv/bin/python -m pip install matplotlib
.venv/bin/python scripts/customer_distribution.py
```

To regenerate charts from the saved counts without rescanning the large file:

```bash
.venv/bin/python scripts/customer_distribution.py --plot-only
```
