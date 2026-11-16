#!/usr/bin/env python3
"""Step 1: sample 1024 calibration prompts (512 conversations, both conditions).

Uses only the Python standard library; no model loading or quantization.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
MARKER = "Additional instruction from the user:"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_source(row):
    source = row.get("source", {})
    return isinstance(source, dict) and all(
        isinstance(source.get(key), str) and source[key].strip()
        for key in ("system_prompt", "user_content")
    )


def sample_calibration(path, num_samples=1024, seed=42):
    """Reservoir-sample N/2 conversations, then emit both conditions for each.

    With-instruction prompts come from the nested unused_instructions entries.
    This dataset has no assistant responses, so calibration uses prompts only.
    """
    if num_samples <= 0 or num_samples % 2:
        raise ValueError("num_samples must be positive and even (two prompts per conversation)")
    rng = random.Random(seed)
    reservoir, seen = [], set()
    eligible = skipped = total = 0
    count = num_samples // 2
    print(f"Streaming calibration source: {path}", flush=True)
    # The source .json is JSONL: one grouped conversation per line.
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            source_id = row.get("source_record_id")
            if not isinstance(source_id, str) or not source_id or not valid_source(row):
                skipped += 1
                continue
            source = row["source"]
            if MARKER in source["system_prompt"] or source_id in seen:
                skipped += 1
                continue
            variants = row.get("unused_instructions")
            variants = variants if isinstance(variants, list) else []
            variants = [v for v in variants
                        if isinstance(v, dict) and valid_source(v)
                        and v["source"]["user_content"] == source["user_content"]
                        and v["source"]["system_prompt"].partition(MARKER)[2].strip()
                        and v["source"]["system_prompt"].partition(MARKER)[0].strip()
                        == source["system_prompt"].strip()]
            if not variants:
                skipped += 1
                continue
            seen.add(source_id)
            eligible += 1
            item = (line_number, row, variants)
            if len(reservoir) < count:
                reservoir.append(item)
            else:
                index = rng.randrange(eligible)
                if index < count:
                    reservoir[index] = item
    if len(reservoir) != count:
        raise ValueError(f"Need {count} eligible conversations; found only {eligible}")
    variant_rng = random.Random(seed)
    samples = []
    for line_number, row, variants in sorted(reservoir, key=lambda item: item[0]):
        variant = variant_rng.choice(variants)
        # Match evaluation's generation_record(..., without_user_instruction=True),
        # including the exact trailing whitespace before the instruction marker.
        without_instruction = {**row, "source": {
            **variant["source"],
            "system_prompt": variant["source"]["system_prompt"].partition(MARKER)[0],
        }}
        for mode, selected in (("without_user_instruction", without_instruction),
                               ("with_user_instruction", variant)):
            source = selected["source"]
            samples.append({
                "source_record_id": row["source_record_id"], "source_line": line_number,
                "record_id": selected.get("record_id"), "user_instruction_mode": mode,
                "messages": [{"role": "system", "content": source["system_prompt"]},
                             {"role": "user", "content": source["user_content"]}],
            })
    random.Random(seed).shuffle(samples)
    stat = path.stat()
    metadata = {
        "source_file": str(path.resolve()), "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns, "seed": seed,
        "source_records": total, "eligible_conversations": eligible,
        "skipped_records": skipped, "sampled_conversations": count, "num_samples": len(samples),
        "condition_counts": dict(Counter(row["user_instruction_mode"] for row in samples)),
        "calibration_content": "system + user prompts, with generation prefix; no assistant responses",
    }
    return samples, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data/query_rewriter_combined_selfins_unused_instructions.json")
    parser.add_argument("--output", type=Path,
                        help="Default: data/quantization/calibration_<N>_seed<S>.jsonl")
    parser.add_argument("--num-samples", type=int, default=1024,
                        help="Total prompts: half with and half without a user instruction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.num_samples % 2:
        parser.error("--num-samples must be positive and even")
    source = args.input.expanduser().resolve()
    output = (args.output or ROOT / "data/quantization" /
              f"calibration_{args.num_samples}_seed{args.seed}.jsonl").expanduser().resolve()
    if source == output:
        parser.error("Output must differ from the source dataset")
    if not source.is_file():
        parser.error(f"Missing calibration source: {source}")
    samples, metadata = sample_calibration(source, args.num_samples, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata.update(calibration_file=str(output), calibration_sha256=sha256(output))
    output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(samples)} prompts from {metadata['sampled_conversations']} conversations "
          f"(half with, half without instructions): {output}", flush=True)


if __name__ == "__main__":
    main()
