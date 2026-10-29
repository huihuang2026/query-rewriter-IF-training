#!/usr/bin/env python3
"""Convert exported data to JSONL, or square-root sample test data from a processed file."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import tempfile
import time


def customer_quotas(counts, sample_size, available=None):
    """Allocate integer quotas proportional to sqrt(count), capped at 10%."""
    if sample_size < 10 or sample_size > sum(counts.values()):
        raise ValueError("Sample size must be at least 10 and no larger than the available records")
    available = counts if available is None else available
    capacities = {customer: min(available.get(customer, 0), sample_size // 10) for customer in counts}
    if sum(capacities.values()) < sample_size:
        raise ValueError("Not enough customers/records to fill the sample with a 10% customer cap")
    weights = {customer: math.sqrt(count) for customer, count in counts.items()}
    low, high = 0.0, float(sample_size)
    for _ in range(80):
        scale = (low + high) / 2
        if sum(min(capacities[c], scale * weights[c]) for c in counts) < sample_size:
            low = scale
        else:
            high = scale
    allocation = {c: min(capacities[c], low * weights[c]) for c in counts}
    quotas = {c: math.floor(value) for c, value in allocation.items()}
    # Largest remainders make the integer quotas add up to exactly sample_size.
    candidates = sorted((c for c in counts if quotas[c] < capacities[c]),
                        key=lambda c: (-(allocation[c] - quotas[c]), c))
    for customer in candidates[:sample_size - sum(quotas.values())]:
        quotas[customer] += 1
    assert sum(quotas.values()) == sample_size
    return quotas


def user_instruction(record):
    """Extract the instruction and normalize whitespace for duplicate detection."""
    prompt = record.get("system_prompt", "")
    marker = "Additional instruction from the user:"
    if not isinstance(prompt, str) or marker not in prompt:
        return ""
    return " ".join(prompt.split(marker, 1)[1].split())


def sample_customers(source, destination, sample_size, seed):
    """Square-root allocate customers while selecting each instruction once globally."""
    if not source.is_file() or source.resolve() == destination.resolve():
        raise ValueError("Sampling requires an existing JSONL input file and a different output file")

    rng = random.Random(seed)
    counts, representatives = Counter(), {}
    skipped = 0
    started = time.monotonic()
    initial_stat = source.stat()
    print("Counting customers and unique user instructions...", flush=True)
    with source.open("rb") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                customer = record["customer_identifier"]
                if not isinstance(customer, str) or not customer.strip():
                    raise ValueError("Missing customer identifier")
                instruction = user_instruction(record)
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"{source}:{index}: {error}") from error
            counts[customer] += 1
            if not instruction:
                skipped += 1
                continue
            # One random representative per instruction, even across customers.
            occurrences = representatives.get(instruction, (0, None, None))[0] + 1
            if rng.randrange(occurrences) == 0:
                representatives[instruction] = (occurrences, customer, index)
            else:
                _, owner, row_index = representatives[instruction]
                representatives[instruction] = (occurrences, owner, row_index)
            if index % 250_000 == 0:
                print(f"Counting: {index:,} lines | {len(representatives):,} unique instructions"
                      f" | {time.monotonic() - started:.0f}s", flush=True)
    print(f"Available: {len(representatives):,} unique instructions"
          f" | Skipped missing/empty instructions: {skipped:,}", flush=True)
    if len(representatives) < sample_size:
        raise ValueError(f"Requested {sample_size:,} records, but only {len(representatives):,} unique instructions exist; lower --sample-size")
    candidates = {customer: [] for customer in counts}
    for _, customer, index in representatives.values():
        candidates[customer].append(index)
    quotas = customer_quotas(counts, sample_size, {c: len(indices) for c, indices in candidates.items()})
    selected = {index for customer, indices in candidates.items()
                for index in rng.sample(indices, quotas[customer])}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=destination.parent,
                                         prefix=destination.name + ".", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            written = 0
            with source.open("rb") as handle:
                for index, line in enumerate(handle, 1):
                    if index in selected:
                        output.write(line.rstrip(b"\r\n") + b"\n")
                        written += 1
                    if index % 250_000 == 0:
                        print(f"Writing: {index:,} lines | Selected: {written:,}/{sample_size:,}", flush=True)
            current_stat = source.stat()
            if written != sample_size or (initial_stat.st_size, initial_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
                raise ValueError("Input changed during sampling; rerun with a stable input")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Sampled {len(selected):,} records into {destination}"
          f" | Largest customer: {max(quotas.values()) / sample_size:.2%}", flush=True)


def read_records(path, on_error=None):
    if path.suffix.lower() == ".json":
        try:
            with path.open(encoding="utf-8") as source:
                data = json.load(source)
        except json.JSONDecodeError:
            # Some exported files contain JSONL despite having a .json extension.
            pass
        else:
            records = data if isinstance(data, list) else [data]
            for index, record in enumerate(records, 1):
                yield index, record
            return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as error:
                    if on_error is None:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
                    on_error(path, line_number, f"invalid outer JSON: {error.msg}", None)


def decode(value, field):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {field}: {error.msg} (character {error.pos})") from error


def content_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if not isinstance(block, dict) or not isinstance(block.get("text"), str):
                raise ValueError("Expected text content blocks")
            texts.append(block["text"])
        return "\n".join(texts)
    raise ValueError("Expected string content or a list of text blocks")


def convert(record):
    if not isinstance(record, dict):
        raise ValueError("Expected an export object")
    messages = decode(record["input"], "input")
    response = decode(record["output"], "output")
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise ValueError("Expected input to contain a list of messages")
    if not isinstance(response, dict):
        raise ValueError("Expected output to contain an assistant message object")
    metadata = record.get("metadata") or {}
    result = {
        "timestamp": record["timestamp"],
        "call_id": metadata.get("call_id") or record.get("thread_identifier"),
        "customer_identifier": record["customer_identifier"],
        "system_prompt": "\n".join(content_text(message["content"]) for message in messages if message.get("role") == "system"),
        "user_content": "\n".join(content_text(message["content"]) for message in messages if message.get("role") == "user"),
        "rewritten_query": content_text(response["content"]),
    }
    for key, value in result.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or empty {key}")
    return result


def report_progress(processed_files, total_files, records, skipped):
    print(f"Files: {processed_files:,}/{total_files:,} ({processed_files / total_files:.1%})"
          f" | Valid records: {records:,} | Skipped: {skipped:,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Export file or directory containing .json/.jsonl files")
    parser.add_argument("output_file", type=Path, help="Combined output JSONL file")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip invalid records and log each error")
    parser.add_argument("--sample", action="store_true", help="Square-root customer sampling with unique user instructions and a 10%% customer cap")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of sampled records (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed (default: 42)")
    args = parser.parse_args()
    source, destination = args.input_path, args.output_file
    if args.sample:
        try:
            sample_customers(source, destination, args.sample_size, args.seed)
        except (OSError, ValueError) as error:
            parser.exit(1, f"Error: {error}\n")
        return
    if source.is_dir():
        files = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in (".json", ".jsonl") and path.resolve() != destination.resolve())
    elif source.is_file():
        files = [source]
    else:
        parser.error(f"Input does not exist: {source}")
    if not files:
        parser.error(f"No .json or .jsonl files found in {source}")
    if any(path.resolve() == destination.resolve() for path in files):
        parser.error("Output must be different from the input file")

    temporary = None
    skipped = 0
    errors_path = destination.with_suffix(".errors.json")
    errors_handle = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.skip_invalid:
            errors_handle = errors_path.open("w", encoding="utf-8")

        def log_error(path, index, error, record):
            nonlocal skipped
            skipped += 1
            entry = {"file": str(path), "line": index, "error": str(error)}
            if isinstance(record, dict):
                entry["record_id"] = record.get("id")
            errors_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            errors_handle.flush()

        count = 0
        report_progress(0, len(files), count, skipped)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            for processed_files, path in enumerate(files, 1):
                for index, record in read_records(path, log_error if args.skip_invalid else None):
                    try:
                        cleaned = convert(record)
                    except (KeyError, ValueError, TypeError, AttributeError) as error:
                        if args.skip_invalid:
                            log_error(path, index, error, record)
                            continue
                        raise ValueError(f"{path}:{index}: {error}") from error
                    output.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    count += 1
                report_progress(processed_files, len(files), count, skipped)
        temporary.replace(destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    finally:
        if errors_handle is not None:
            errors_handle.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Processed {count:,} records from {len(files):,} files into {destination}")
    if args.skip_invalid:
        print(f"Skipped {skipped:,} invalid records. Error log: {errors_path}")


if __name__ == "__main__":
    main()
