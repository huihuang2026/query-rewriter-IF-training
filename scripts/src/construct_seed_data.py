#!/usr/bin/env python3
"""Rebuild customer-capped training pairs from authentic exported instructions.

The shell selects up to ten records per customer directly from cleaned exported data,
generates paired Gemini rewrites, and processes the pairs into SFT examples.
All files use JSON Lines, including .json outputs. The optional sample command
supports historical per-instruction sampling; the shell does not use it.
Gemini is called only by the generate stage.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import random
import tempfile
import threading
import time

from generation_utils import QUERY_FIELDS, positive, request_with_retries
from prepare_test_data import user_instruction

MARKER = "Additional instruction from the user:"



def record_key(row):
    source = row.get("source", {key: value for key, value in row.items()
                                if key not in (*QUERY_FIELDS, "generation_model", "generation_error")})
    return hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def records(path, limit):
    count = 0
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if limit and count >= limit:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as error:
                raise ValueError(f"{path}:{index}: {error}") from error
            count += 1
            yield row


def generation_inputs(row):
    source = row.get("source", row)
    if not isinstance(source, dict):
        raise ValueError("Expected a conversation object")
    full_prompt, conversation = source.get("system_prompt"), source.get("user_content")
    if not all(isinstance(value, str) and value.strip() for value in (full_prompt, conversation)):
        raise ValueError("Expected nonempty system_prompt and user_content")
    base_prompt, separator, instruction = full_prompt.partition(MARKER)
    if not separator or not base_prompt.strip() or not instruction.strip():
        raise ValueError("Expected base instructions followed by a nonempty user-defined instruction")
    return full_prompt, base_prompt.rstrip(), conversation


def generate(args, *, input_reader=records, prompt_reader=generation_inputs):
    from google import genai
    from google.genai import types
    from tqdm import tqdm

    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("Set GEMINI_API_KEY before running")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if args.resume and args.output.exists():
        previous = {record_key(row): row for row in records(args.output, None)}
        if any(row.get("generation_model") != args.model for row in previous.values()):
            raise ValueError("Resume model differs from saved results; use a different output file")
        print(f"Resuming with {len(previous):,} saved records", flush=True)
    state, clients, lock = threading.local(), [], threading.Lock()

    def pending_rows():
        for row in input_reader(args.input, args.limit):
            saved = previous.get(record_key(row))
            if args.resume and saved:
                if complete(saved):
                    continue
                row = {**row, **{field: saved[field] for field in QUERY_FIELDS if isinstance(saved.get(field), str)}}
            yield row

    def rewrite(row):
        # Preserve the source; never send its previous rewritten query to Gemini.
        result = {**row, "generation_model": args.model}
        result.pop("generation_error", None)
        if not args.resume:
            for field in QUERY_FIELDS:
                result.pop(field, None)
        try:
            full_prompt, base_prompt, conversation = prompt_reader(row)
            if not hasattr(state, "client"):
                state.client = genai.Client(api_key=key, vertexai=False,
                                            http_options=types.HttpOptions(timeout=120_000))
                with lock:
                    clients.append(state.client)
            for field, prompt in (("query_with_user_instruction", full_prompt),
                                  ("query_without_user_instruction", base_prompt)):
                if args.resume and isinstance(result.get(field), str):
                    continue
                # Independent requests: neither rewrite is exposed to the other.
                response = request_with_retries(state.client,
                    model=args.model, contents=conversation,
                    config=types.GenerateContentConfig(system_instruction=prompt,
                             thinking_config=types.ThinkingConfig(thinking_level=args.thinking_level),
                             max_output_tokens=args.max_output_tokens)
                )
                # A normal STOP with no text can follow an instruction that
                # excludes the current request (e.g. booking/cancellation).
                query = response.text or ""
                if not isinstance(query, str):
                    raise ValueError(f"{field}: Gemini returned invalid query text")
                result[field] = query.strip()
        except Exception as error:
            result["generation_error"] = f"{field}: {error}" if 'field' in locals() else str(error)
        return result

    completed = sum(all(isinstance(row.get(field), str) for field in QUERY_FIELDS) for row in previous.values())
    started = time.monotonic()
    try:
        pending = pending_rows()
        with ThreadPoolExecutor(max_workers=args.workers) as executor, args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
            batch_number = 0
            while batch := list(islice(pending, args.batch_size)):
                batch_number += 1
                batch_started = time.monotonic()
                results = [None] * len(batch)
                with tqdm(total=len(batch), desc=f"Generation batch {batch_number}", unit="pair") as progress:
                    futures = {executor.submit(rewrite, row): index for index, row in enumerate(batch)}
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()
                        progress.update(1)
                for result in results:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    previous[record_key(result)] = result
                handle.flush()
                errors = sum("generation_error" in row for row in results)
                empty = sum(row.get(field) == "" for row in results for field in
                            ("query_with_user_instruction", "query_without_user_instruction"))
                completed += len(batch)
                print(f"Completed batch {batch_number}: {len(batch)} pairs, {errors} errors, {empty} empty rewrites"
                      f" | Batch: {time.monotonic() - batch_started:.1f}s"
                      f" | Total: {completed:,} records, {time.monotonic() - started:.1f}s", flush=True)
                for index, row in enumerate(results):
                    if row.get("generation_error"):
                        print(f"Batch {batch_number}, pair {index + 1}: {row['generation_error']}", flush=True)
                if errors == len(batch):
                    raise ValueError(f"Every pair in the batch failed; stopping to avoid repeated failures. Results saved to {args.output}")
    finally:
        for client in clients:
            client.close()
        # Replace retried partial records in place and keep one row per input.
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in previous.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    failed = sum(bool(row.get("generation_error")) for row in previous.values())
    print(f"Saved {len(previous):,} records: {len(previous) - failed:,} completed, {failed:,} errors: {args.output}")
    if failed:
        print("Run again with --resume to retry failed variants; completed rewrites are preserved.")


def rows(path):
    with path.open("rb") as handle:
        for index, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Expected a JSON object")
                except ValueError as error:
                    raise ValueError(f"{path}:{index}: {error}") from error
                yield index, line, row


def sample_exported(source, test, destination, seed=42, max_per_instruction=5):
    if max_per_instruction < 1:
        raise ValueError("max_per_instruction must be greater than zero")
    if destination.resolve() in {source.resolve(), test.resolve()}:
        raise ValueError("Output must differ from both source and test files")
    blocked = set()
    test_conversations = set()
    for _, _, row in rows(test):
        test_source = row.get("source", row)
        instruction = user_instruction(test_source)
        if not instruction:
            raise ValueError("Test record has no user-defined instruction; cannot guarantee exclusion")
        blocked.add(instruction)
        test_conversations.add(" ".join(test_source["user_content"].split()))
    print(f"Test instructions to exclude: {len(blocked):,}", flush=True)

    rng = random.Random(seed)
    selected, excluded, seen_examples = {}, set(), set()
    total = missing = excluded_records = duplicates = 0
    initial_stat = source.stat()
    for index, line, row in rows(source):
        total += 1
        data = row.get("source", row)
        instruction = user_instruction(data)
        if not instruction:
            missing += 1
        elif instruction in blocked or " ".join(data.get("user_content", "").split()) in test_conversations:
            if instruction in blocked:
                excluded.add(instruction)
            excluded_records += 1
        else:
            # Ignore export timestamps/call IDs when deduplicating training pairs.
            example = (data.get("system_prompt"), data.get("user_content"),
                       data.get("rewritten_query", data.get("gemini_rewritten_query")))
            signature = hashlib.sha256(json.dumps(example, ensure_ascii=False).encode()).digest()
            if signature in seen_examples:
                duplicates += 1
            else:
                seen_examples.add(signature)
                count, reservoir = selected.setdefault(instruction, (0, []))
                count += 1
                if len(reservoir) < max_per_instruction:
                    reservoir.append((index, line))
                else:
                    position = rng.randrange(count)
                    if position < max_per_instruction:
                        reservoir[position] = (index, line)
                selected[instruction] = count, reservoir
        if total % 250_000 == 0:
            print(f"Processed: {total:,} | Training instructions: {len(selected):,}", flush=True)
    current_stat = source.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
        raise ValueError("Source changed during processing; rerun with a stable input")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            chosen = sorted(item for _, reservoir in selected.values() for item in reservoir)
            for _, line in chosen:
                output.write(line.rstrip(b"\r\n") + b"\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Source records: {total:,}")
    print(f"Unique instructions before test exclusion: {len(selected) + len(excluded):,}")
    print(f"Excluded test instructions: {len(excluded):,}; records excluded by instruction/conversation: {excluded_records:,}")
    print(f"Skipped records without instructions: {missing:,}")
    print(f"Duplicate training pairs skipped: {duplicates:,}")
    print(f"Training records: {len(chosen):,} | Unique instructions: {len(selected):,}"
          f" | Maximum records per instruction: {max_per_instruction}")
    print(f"Output: {destination}")


def source(row):
    return row.get("source", row)


def read(path, snapshot=False):
    # The reuse file may still be growing: read only complete lines in this snapshot.
    with path.open("rb") as handle:
        size = path.stat().st_size
        while handle.tell() < size:
            line = handle.readline()
            if snapshot and (handle.tell() > size or not line.endswith(b"\n")):
                break
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object in {path}")
                yield row


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def identity(row):
    data = source(row)
    return data.get("customer_identifier"), user_instruction(data)


def complete(row):
    return not row.get("generation_error") and all(isinstance(row.get(k), str) for k in QUERY_FIELDS)


def prepare(args):
    protected = {p.resolve() for p in (args.input, args.test, args.reuse) if p is not None}
    if args.output.resolve() in protected or args.selected.resolve() in protected or args.output.resolve() == args.selected.resolve():
        raise ValueError("Output and selected-input paths must be separate from all source files")
    blocked = set()
    test_pairs = set()
    test_conversations = set()
    for row in read(args.test):
        instruction = user_instruction(source(row))
        if not instruction:
            raise ValueError("Test record has no additional instruction; cannot guarantee exclusion")
        blocked.add(instruction)
        data = source(row)
        test_pairs.add((data.get("system_prompt"), data.get("user_content")))
        test_conversations.add(" ".join(data["user_content"].split()))

    candidates = defaultdict(dict)
    for row in read(args.input):
        data = source(row)
        org, instruction = identity(row)
        if not org or not instruction or instruction in blocked:
            continue
        if (data.get("system_prompt"), data.get("user_content")) in test_pairs:
            continue
        if " ".join(data.get("user_content", "").split()) in test_conversations:
            continue
        try:
            generation_inputs(row)
        except ValueError:
            continue
        candidates[org].setdefault(record_key(row), row)

    reused = {}
    saved = {}
    # Completed records in this batch's output take priority over the live reuse file.
    for path in (args.reuse, args.output):
        if path is None or not path.exists():
            continue
        for row in read(path, snapshot=True):
            org, instruction = identity(row)
            key = record_key(row)
            if key not in candidates.get(org, {}):
                continue
            if row.get("generation_model") != args.model:
                continue
            try:
                generation_inputs(row)
            except ValueError:
                continue
            saved[key] = row
            if complete(row):
                reused[key] = row

    if args.selected.exists():
        chosen = list(read(args.selected))
        print(f"Reusing fixed selection: {args.selected}")
    else:
        rng = random.Random(args.seed)
        chosen = []
        for org in sorted(candidates):
            available = sorted(candidates[org])
            ready = [key for key in available if key in reused]
            missing = [key for key in available if key not in reused]
            selected = rng.sample(ready, min(args.max_per_customer, len(ready)))
            selected += rng.sample(missing, min(args.max_per_customer - len(selected), len(missing)))
            for key in selected:
                row = reused.get(key, candidates[org][key])
                # Preserve the generated record's identity, but strip generated fields from input.
                chosen.append({k: v for k, v in row.items()
                               if k not in (*QUERY_FIELDS, "generation_model", "generation_error")})

    seen = defaultdict(set)
    seeded = []
    for row in chosen:
        org, instruction = identity(row)
        key = record_key(row)
        if key not in candidates.get(org, {}) or key in seen[org]:
            raise ValueError("Selected file contains an excluded, unavailable, or repeated record")
        seen[org].add(key)
        if len(seen[org]) > args.max_per_customer:
            raise ValueError("Selected file exceeds customer cap; use a new output name when changing the cap")
        prior = saved.get(record_key(row))
        if prior:
            seeded.append(prior)
    if not chosen:
        raise ValueError("No eligible records after test exclusion")
    write(args.selected, chosen)
    write(args.output, seeded)
    ready_count = sum(complete(row) for row in seeded)
    print(f"Excluded test instructions: {len(blocked):,}")
    print(f"Customers: {len(seen):,} | Selected records: {len(chosen):,}")
    print(f"Reused completed pairs: {ready_count:,} | Pairs needing generation: {len(chosen) - ready_count:,}")
    print(f"Customers with fewer than {args.max_per_customer} eligible records: "
          f"{sum(len(v) < args.max_per_customer for v in seen.values()):,}")
    print(f"Selected input: {args.selected}\nPair output: {args.output}")
    return len(chosen) - ready_count


def read_records(path):
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            yield from json.load(handle)
        else:
            for line in handle:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield None  # Count an invalid JSONL record, then continue.


def convert(row, counts):
    if not isinstance(row, dict):
        counts["invalid_records"] += 1
        return
    source = row.get("source", row)
    if not isinstance(source, dict) or row.get("generation_error") or source.get("generation_error"):
        counts["invalid_records"] += 1
        return
    system = source.get("system_prompt_with_user_instruction", source.get("system_prompt"))
    conversation = source.get("user_content")
    if not all(isinstance(value, str) and value.strip() for value in (system, conversation)):
        counts["invalid_records"] += 1
        return
    base, marker, instruction = system.partition(MARKER)
    if not marker or not base.strip() or not instruction.strip():
        counts["invalid_records"] += 1
        return
    if ("system_prompt_without_user_instruction" in source
            and source["system_prompt_without_user_instruction"] != base):
        counts["invalid_records"] += 1
        return
    counts["valid_source_records"] += 1
    for field, prompt in (("query_with_user_instruction", system),
                          ("query_without_user_instruction", base)):
        target = row.get(field, source.get(field))
        if target is None or (isinstance(target, str) and target.strip() in ("", "[NO_QUERY]")):
            counts["removed_empty_queries"] += 1
            continue
        if not isinstance(target, str):
            counts["invalid_query_fields"] += 1
            continue
        yield {"prompt": [{"role": "system", "content": prompt},
                          {"role": "user", "content": conversation}],
               "completion": [{"role": "assistant", "content": target}]}


def process(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must be different files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in read_records(source):
                counts["source_records"] += 1
                for example in convert(row, counts):
                    handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                    counts["training_examples"] += 1
            if not counts["training_examples"]:
                raise ValueError(f"No nonempty training examples: {dict(counts)}")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    for name in ("source_records", "valid_source_records", "invalid_records",
                 "removed_empty_queries", "invalid_query_fields", "training_examples"):
        print(f"{name}: {counts[name]:,}")
    print(f"Processed training data: {destination}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Sample exported data and exclude test instructions/conversations")
    sample.add_argument("--input", type=Path, required=True)
    sample.add_argument("--test", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--max-per-instruction", type=positive, default=5)
    sample.add_argument("--seed", type=int, default=42)

    pairs = commands.add_parser("generate", help="Cap records by customer, reuse pairs, and generate missing rewrites")
    pairs.add_argument("--input", type=Path, required=True)
    pairs.add_argument("--test", type=Path, required=True)
    pairs.add_argument("--reuse", type=Path,
                       help="Optional existing paired Gemini output; a missing file is ignored")
    pairs.add_argument("--output", type=Path, required=True)
    pairs.add_argument("--max-per-customer", type=positive, default=10)
    pairs.add_argument("--seed", type=int, default=42)
    pairs.add_argument("--model", default="gemini-3.8-flash")
    pairs.add_argument("--thinking-level", choices=("low", "medium", "high"), default="low")
    pairs.add_argument("--batch-size", type=positive, default=100)
    pairs.add_argument("--workers", type=positive, default=20)
    pairs.add_argument("--max-output-tokens", type=positive, default=8192)
    pairs.add_argument("--prepare-only", action="store_true", help="Select/reuse data without calling Gemini")

    processed = commands.add_parser("process", help="Convert matched branches into SFT prompt/completion examples")
    processed.add_argument("--input", type=Path, required=True)
    processed.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "sample":
            sample_exported(args.input, args.test, args.output, args.seed, args.max_per_instruction)
        elif args.command == "generate":
            args.selected = args.output.with_name(args.output.stem + "_input.json")
            pending = prepare(args)
            if pending and not args.prepare_only:
                args.input, args.resume, args.limit = args.selected, True, None
                generate(args)
        else:
            process(args.input, args.output)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Saved pairs will be reused on the next run.\n")


if __name__ == "__main__":
    main()
