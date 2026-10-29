# Self-instruct query-rewriter data construction

Status: plan for review only. No API generation or training has started.

## Objective

Use conversations from the previous dataset to create new training examples with
additional user-defined rewriting instructions. Create up to 10 distinct, valid
instruction variants per selected conversation.

Keep the existing base system prompt unchanged. Append each new instruction under
the existing `Additional instruction from the user:` marker.

## 1. Extract the seed instructions

Read:

`data/export-data-train/query_rewriter_training_customer_duplicated_processed.json`

This file currently contains:

- 2,806 examples with an additional instruction.
- 2,857 examples without an additional instruction.
- 725 distinct additional instruction texts after whitespace normalization.
- One distinct base system prompt.

Extract the additional instruction text as seed examples, and separately retain
the exact base prompt before the marker. The generation unit is a conversation,
not a unique base prompt; otherwise there would be only one generation unit.

Save the seed pool for review before the pilot. Remove obvious parameter-only
duplicates from the seed-example sampling pool, such as device-model and timestamp
variations of the same rule. This grouping affects seed sampling only, not original
instruction text or existing training files. Select a small, varied set of seeds
for each request rather than putting all 725 into every prompt.

## 2. Select previous-data conversations

Read:

`data/Query-Rewriter-Dataset/query_rewriter_gemini_annotated_cleaned.jsonl`

- Sample conversations randomly with a fixed seed and an adjustable sample count.
- Deduplicate normalized conversation text, and exclude matches against the
  current training conversations and the held-out test conversations.
- Keep original conversation text unchanged in the generated examples.

The old dataset has no user-defined instruction. Its conversations will receive
the exact base prompt extracted in step 1. Its existing rewritten query is not
automatically a valid target once we add a new constraint.

Proposed pilot: **100 eligible conversations × 10 variants = up to 1,000 examples**.
The number of conversations remains a shell parameter. Do not start with all
407,319 old records, which would imply over four million candidate variants.

## 3. Generate new additional instructions

For each conversation, provide Gemini with:

1. The unchanged base query-rewriting instruction.
2. The conversation.
3. A few diverse seed additional instructions, explicitly labeled as examples.
4. A request to propose 10 different, applicable rewriting constraints.

One request can return the 10 candidates as a JSON list. This is equivalent to
10 variants per conversation without requiring 10 independent invention calls.
Run different conversations concurrently using batch size and worker parameters.

Each candidate returns only a `user_defined_instruction` string. 

Seeds illustrate possible behaviors; they must not be blindly copied onto an
unrelated conversation. Generation must not receive held-out test instructions.

## 4. Generate and check the rewritten queries

For each accepted instruction:

```text
system = unchanged base prompt + existing additional-instruction marker + new instruction
user = unchanged conversation
assistant target = a newly generated rewritten query
```

Use an independent Gemini request to generate the target. Do not reuse the old
query as the constrained target and do not expose sibling targets to generation.


## 5. Files and execution

Put all new data under:

```text
data/export-data-self-instruct/
  seed_instructions.jsonl
  conversations_sample_100.jsonl
  instruction_candidates_sample_100.jsonl
  rewrite_candidates_sample_100.jsonl
  training_processed_sample_100.jsonl
```

Intermediate records preserve a source identifier, original conversation, base
prompt, new instruction, generated query, and review. Final training data uses
the existing `prompt`/`completion` format and contains no review metadata in the
messages presented to the model.

After approval, implement one Python entry point with separate preparation,
instruction generation, rewriting/review, and processing commands, plus one simple
shell launcher. Expose conversation count, variants per conversation (default 10),
model, batch size, workers, and random seed near the top of the shell script.

Reuse the existing Gemini client, retries, progress reporting, and resume behavior.
Persist completed stages and retry only missing work. Avoid modifying any current
training data, checkpoints, or evaluation files.

## Review checkpoints

1. Review the extracted seed pool and selected pilot conversations.
2. Generate and manually inspect a small set of constraints before the full pilot.
3. Review constrained rewrites before accepting the training export.
4. Decide the training mixture with existing constrained and unconstrained data
   after measuring pilot quality and diversity.

Scope assumption for approval: use the **previous dataset for conversations** and
the **current processed dataset for seed instructions and the base prompt**. If
the intended conversations are instead the current dataset's base-only examples,
change step 2; the rest of the workflow stays the same.
