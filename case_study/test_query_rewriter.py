#!/usr/bin/env python3
"""Compare query rewrites by context length or terminology instruction."""

import argparse
import csv
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

case_dir = Path(__file__).resolve().parent
workspace = case_dir.parent
default_model_path = workspace / "models/Query-Rewrite-Qwen3-4B-2507"
input_path = case_dir / "data/retrieval_issue_testset_no_analysis.json"

parser = argparse.ArgumentParser(
    prog="test_query_rewriter.sh",
    description="Compare original-prompt recent 10 turns vs full conversation for context cases, and original vs "
                "terminology-augmented prompts at 10 turns for terminology cases.",
    epilog="Appends model columns to output/retrieval_issue_testset_no_analysis.csv. "
           "Repeated model runs overwrite that model's columns. JSONL details are saved per model. "
           "PYTHON_BIN and GPU environment variables select the Python interpreter and GPU.")
parser.add_argument("--model-path", type=Path, default=default_model_path,
    help="Local model directory (default: models/Query-Rewrite-Qwen3-4B-2507).")
parser.add_argument("--model-name", help="Model label for the CSV columns and JSONL details.")
parser.add_argument("--lora-path", type=Path, help="Optional trained LoRA adapter directory.")
args = parser.parse_args()
model_path = args.model_path
model_name = args.model_name or model_path.name
csv_output = case_dir / "output" / f"{input_path.stem}.csv"
json_output = case_dir / "output" / f"{input_path.stem}_{model_name}.json"

# Exact common system prompt from the available case-study rewrite logs.
# The shared generator supplies the model's chat template and decoding settings.
SYSTEM_PROMPT = """You are an expert query-rewriter in a Retrieval-Augmented Generation (RAG) system.
Based on the latest conversation context, write a brief, standalone query ideal for semantic search.

Guidelines:
1. Start with the customer's most recent message.
2. If earlier turns contain concrete details or topics the latest message needs (e.g., names, dates, file titles, pronoun antecedents), copy only those details.
3. Do **not** add information that is not explicitly written in the chat history.
4. Preserve the original language — never translate.
5. Remove conversational fillers and disambiguate pronouns.
6. If the latest message is already a clear standalone query, return it unchanged.
7. When the agent asks diagnostic or clarifying questions and the customer provides answers, consolidate the key information from both sides into a coherent query. Extract the topic/subject from the agent's question and combine it with the customer's concrete answer.
8. For multi-turn Q&A sequences, accumulate all confirmed facts the customer has provided into a single comprehensive query that captures the full context.
9. If the customer's response is a brief confirmation, negation, or short answer, incorporate the agent's question context to form a meaningful standalone statement.
10. When the agent introduces specific named entities, proper nouns, product names, series names, categories, or enumerated options, and the customer's request relates to them, preserve those exact terms in the rewritten query rather than generalizing to broader categories.

Return **only** the rewritten query — no extra text, tags, or explanations."""


ADDITIONAL_INSTRUCTION_MARKER = "Additional instruction from the user:"

# User-defined instruction for this case study. Edit the terminology here.
USER_DEFINED_INSTRUCTION = """When the customer's description matches one of the following professional fault terms, use that exact term as the issue type in the rewritten query.

Professional fault terms:
- Abnormal data transmission when connecting to computer
- Can detect WiFi, but unable to connect to mobile APP
- Can detect bluetooth but unable to connect to it
- Cannot connect to the App
- File Missing
- Prompt to format when connected to computer
- Slow download speed
- Unable to Detect Wi-Fi Signal
- Unable to Recognize Camera when Connected to Computer
- Unable to detect Bluetooth
- WiFi disconnect with app
- Cannot connect to the Action Pod
- Selfie Stick not invisible
- Stitching Issues at Dive Case Bottom
- Filament in lens
- Fog (Lens/Lens Guards)
- Lens Crack/Break
- Lens Guards Crack
- Lens/Lens Guards Scratch Damage
- Screen Crack
- dust in lens
- white dot in lens
- Audio Noise
- Battery Drain
- Black screen
- Button not working
- Camera auto restart
- Cannot Fully Charge
- Charge Slow
- Crash
- Liquid ingress into camera
- No sound
- Overheat
- Rapid Battery Drain/High Power Consumption
- Shaking sound
- Stop Recording Automatically
- Unable to charge
- camera auto shut-off
- camera cannot power on
- Abnormal Color Balance/Temperature
- Black Shadow at the Seam
- Black Spot/White Spot/Dead Pixel
- Bright Seam / White Lines at Stitching
- Exposure Issue
- Flicker
- Footage review with mosaic pixelation
- Image Black and White
- Image Distortion
- Image Not Leveled
- Lens Flare
- Motion Blur
- Poor Stabilization
- Poor image quality
- Screen Flicker
- Splashscreen
- Stitching Ghost / Blur
- Stitching Line Fluctuate
- Stitching Misalignment / Discontinuity
- chromatic Aberration
- stripe/band of screen"""


def prepare_records(rows):
    augmented_prompt = SYSTEM_PROMPT + "\n\n" + ADDITIONAL_INSTRUCTION_MARKER + "\n" + USER_DEFINED_INSTRUCTION
    prepared = []
    for index, row in enumerate(rows, 1):
        conversation = row["conversation"]
        if not conversation or conversation[-1].get("role") != "user":
            raise ValueError(f"Sample {index}: conversation must end with a user response")
        for turn in conversation:
            if turn.get("role") not in {"agent", "user"} or not isinstance(turn.get("content"), str):
                raise ValueError(f"Sample {index}: expected user/agent turns with text content")
        category = row["rewriting_correction"]["issure_category"]
        if category in {"out_ouf_context", "out_of_context"}:
            comparison = "context_length"
            variants = (("original_instruction_10_turns", 10, SYSTEM_PROMPT),
                        ("original_instruction_full_conversation", None, SYSTEM_PROMPT))
        elif category == "terminology_mismatch":
            comparison = "terminology_instruction"
            variants = (("original_instruction_10_turns", 10, SYSTEM_PROMPT),
                        ("with_terminology_instruction_10_turns", 10, augmented_prompt))
        else:
            raise ValueError(f"Sample {index}: unsupported issue category {category!r}")
        for condition, turn_limit, system_prompt in variants:
            selected = conversation[-turn_limit:] if turn_limit is not None else conversation
            user_content = "\n".join(
                f"{'customer' if turn['role'] == 'user' else 'agent'}: {turn['content']}"
                for turn in selected)
            # The user message contains only dialogue. Review labels and proposed
            # corrections stay in the output; the shared terminology guide is
            # independent of each sample's expected rewrite.
            prepared.append({
                "record_id": f"sample_{index}:{condition}",
                "sample_index": index,
                "comparison_type": comparison,
                "condition": condition,
                "context_turn_limit": turn_limit,
                "context_turn_count": len(selected),
                "source": {"system_prompt": system_prompt, "user_content": user_content},
            })
    return prepared


def arrange_comparison_table(columns, table):
    # Collapse old numbered columns into the model's main columns, keeping the latest run.
    selected = {}
    for column in columns:
        match = re.fullmatch(r"((?:Rewrite|Contrast rewrite) \(.+\)) \[run (\d+)\]", column)
        name, run = (match[1], int(match[2])) if match else (column, 1)
        if name not in selected or run >= selected[name][0]:
            selected[name] = (run, column)
    arranged = [{name: row[source] for name, (_, source) in selected.items()} for row in table]
    for row in arranged:
        if row["Issue type"] == "Terminology mismatch":
            row["Contrast method"] = USER_DEFINED_INSTRUCTION
    arranged.sort(key=lambda row: row["Issue type"] != "Out of context")
    return list(selected), arranged


def write_comparison_csv(rows, path):
    if not rows:
        raise ValueError("No model results to write")
    models = {row["generation_model"] for row in rows}
    if len(models) != 1:
        raise ValueError("Write results from one model at a time")
    model = next(iter(models))
    base_columns = ("Sample", "Issue type", "Original conversation", "Original Rewrite",
                    "Expected Rewrite", "Contrast method")
    table = []
    queries = {}
    for index, row in enumerate(rows, 1):
        category = row["rewriting_correction"]["issure_category"]
        if category in {"out_ouf_context", "out_of_context"}:
            issue_type = "Out of context"
            contrast_condition = "original_instruction_full_conversation"
            contrast_method = "Full conversation + original instruction"
        elif category == "terminology_mismatch":
            issue_type = "Terminology mismatch"
            contrast_condition = "with_terminology_instruction_10_turns"
            contrast_method = USER_DEFINED_INSTRUCTION
        else:
            raise ValueError(f"Sample {index}: unsupported issue category {category!r}")
        rewrites = row["generated_rewrites"]
        baseline = rewrites["original_instruction_10_turns"]["generated_query"]
        contrast = rewrites[contrast_condition]["generated_query"]
        if not all(isinstance(query, str) and query.strip() for query in (baseline, contrast)):
            raise ValueError(f"Sample {index}: missing rewrite for CSV comparison")
        conversation = "\n".join(
            f"{turn.get('turn', number)}. {turn['role'].capitalize()}: {turn['content'].strip()}"
            for number, turn in enumerate(row["conversation"], 1))
        table.append({
            "Sample": str(index),
            "Issue type": issue_type,
            "Original conversation": conversation,
            "Original Rewrite": row["rewriting_correction"]["original_rewrite"] or "",
            "Expected Rewrite": row["rewriting_correction"]["expected_rewrite"],
            "Contrast method": contrast_method,
        })
        queries[str(index)] = (baseline, contrast)

    path.parent.mkdir(parents=True, exist_ok=True)
    # Serialize the read/update/write step when model runs finish concurrently.
    with path.with_name(f".{path.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        columns = list(base_columns)
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                columns = list(reader.fieldnames or [])
                existing = list(reader)
            if not set(base_columns).issubset(columns) or len(columns) != len(set(columns)):
                raise ValueError("The shared CSV has missing or duplicate columns")
            existing_by_sample = {row["Sample"]: row for row in existing}
            if (len(existing_by_sample) != len(existing)
                    or set(existing_by_sample) != set(queries)):
                raise ValueError("The shared CSV contains a different set of samples")
            for row in table:
                saved = existing_by_sample[row["Sample"]]
                if any(saved[key] != row[key] for key in
                       ("Issue type", "Original conversation")):
                    raise ValueError(f"Sample {row['Sample']} differs from the shared CSV")
            # Preserve existing model results, reference rewrites and reviewer edits.
            table = existing

        columns, table = arrange_comparison_table(columns, table)
        rewrite_column = f"Rewrite ({model})"
        contrast_column = f"Contrast rewrite ({model})"
        for column in (rewrite_column, contrast_column):
            if column not in columns:
                columns.append(column)
        for row in table:
            row[rewrite_column], row[contrast_column] = queries[row["Sample"]]

        # Quote multiline conversations as single cells; preserve Unicode for spreadsheets.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_output = Path(stream.name)
            try:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(table)
                stream.flush()
                temporary_output.replace(path)
            finally:
                temporary_output.unlink(missing_ok=True)
    return rewrite_column, contrast_column


def main():
    if not (model_path / "config.json").is_file():
        raise ValueError(f"Local model not found: {model_path}")
    if args.lora_path is not None and not all((args.lora_path / filename).is_file()
            for filename in ("adapter_config.json", "adapter_model.safetensors")):
        raise ValueError(f"LoRA adapter files not found: {args.lora_path}")
    if input_path.resolve() in {json_output.resolve(), csv_output.resolve()}:
        raise ValueError("Output must differ from the input dataset")
    with input_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("No samples in the input dataset")
    prepared = prepare_records(rows)
    print(f"Model: {model_name} | weights: {model_path}", flush=True)
    if args.lora_path is not None:
        print(f"LoRA adapter: {args.lora_path}", flush=True)
    print(f"Samples: {len(rows)} | rewrites: {len(prepared)}", flush=True)
    print("Context cases: original instruction, recent 10 turns vs full conversation.", flush=True)
    print("Terminology cases: recent 10 turns, original vs original + user instruction.", flush=True)
    print("Terminology instruction: USER_DEFINED_INSTRUCTION in test_query_rewriter.py", flush=True)
    print("Context length counts stored user/agent entries, not pairs of messages.", flush=True)
    print(f"CSV comparison: {csv_output}", flush=True)
    print(f"JSONL details: {json_output}", flush=True)

    helper_path = workspace / "scripts/src/test_query_rewriter.py"
    spec = importlib.util.spec_from_file_location("query_rewriter_inference", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    with tempfile.TemporaryDirectory(prefix="case-study-rewrites-") as directory:
        temporary = Path(directory)
        subset, generated = temporary / "input.jsonl", temporary / "generated.jsonl"
        subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prepared),
            encoding="utf-8")
        # Same generation function, temperature=0, chat template, thinking setting
        # and 256-token output limit used by scripts/test_query_rewriter.sh.
        helper.generate(SimpleNamespace(input=subset, output=generated, limit=None,
            model_path=model_path, model_name=model_name, lora_path=args.lora_path,
            max_new_tokens=256, both_instruction_conditions=False,
            without_user_instruction=False))
        results = [json.loads(line) for line in generated.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(results) != len(prepared):
        raise ValueError("Generated result count differs from the number of prompts")

    output_rows = [{**row, "generation_model": model_name, "generated_rewrites": {}}
        for row in rows]
    for request, result in zip(prepared, results):
        if result["record_id"] != request["record_id"]:
            raise ValueError("Generated record order differs from the input")
        query = result["generated_query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Empty generated rewrite: {request['record_id']}")
        output_row = output_rows[request["sample_index"] - 1]
        output_row["comparison_type"] = request["comparison_type"]
        output_row["generated_rewrites"][request["condition"]] = {
            "generated_query": query,
            "context_turn_limit": request["context_turn_limit"],
            "context_turn_count": request["context_turn_count"],
            "user_instruction_mode": result["user_instruction_mode"],
            "system_prompt": request["source"]["system_prompt"],
            "user_content": request["source"]["user_content"],
        }
        print(f"[{request['sample_index']}/{len(rows)}] {request['condition']}: {query}", flush=True)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    # Keep any previous result intact if inference or output validation fails.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=json_output.parent,
            prefix=f".{json_output.name}.", suffix=".tmp", delete=False) as stream:
        temporary_output = Path(stream.name)
        try:
            for row in output_rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            temporary_output.replace(json_output)
        finally:
            temporary_output.unlink(missing_ok=True)
    model_columns = write_comparison_csv(output_rows, csv_output)
    print(f"Updated {len(output_rows)} shared CSV rows: {csv_output}")
    print(f"Updated model columns: {', '.join(model_columns)}")
    print(f"Saved {len(output_rows)} JSONL detail records: {json_output}")


if __name__ == "__main__":
    main()
