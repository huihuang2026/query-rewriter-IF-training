#!/usr/bin/env python3
"""Render the held-out test set for vLLM's HTTP speed benchmark; no inference/judging."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/src"))
from test_query_rewriter import (  # Same prompt construction as the quality evaluation.
    additional_instruction, generation_record, messages, records,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(args):
    from transformers import AutoTokenizer

    source, output = args.input.resolve(), args.output.resolve()
    if source == output:
        raise ValueError("Benchmark output must differ from the test set")
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rows = list(records(source, None))
    if not rows:
        raise ValueError("Empty test set")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, lengths = Counter(), []
    with tempfile.NamedTemporaryFile("w", dir=output.parent, encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                if not additional_instruction(row["source"]["system_prompt"]):
                    raise ValueError("Both-condition benchmarking requires a user instruction in each test row")
                for without_instruction in (True, False):
                    branch = generation_record(row, without_instruction)
                    prompt = tokenizer.apply_chat_template(
                        messages(branch), tokenize=False, add_generation_prompt=True, enable_thinking=False,
                    )
                    lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
                    counts[branch["user_instruction_mode"]] += 1
                    handle.write(json.dumps({
                        "prompt": prompt, "output_tokens": args.max_new_tokens,
                        "record_id": row.get("record_id"),
                        "user_instruction_mode": branch["user_instruction_mode"],
                    }, ensure_ascii=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(output)
    metadata = {
        "test_file": str(source), "test_sha256": sha256(source),
        "benchmark_file": str(output), "benchmark_sha256": sha256(output),
        "tokenizer": str(args.tokenizer.resolve()),
        "source_conversations": len(rows), "num_prompts": len(lengths),
        "condition_counts": dict(counts),
        "chat_template_kwargs": {"add_generation_prompt": True, "enable_thinking": False},
        "max_new_tokens": args.max_new_tokens,
        "input_tokens": {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)},
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Prepared {len(lengths)} benchmark prompts from {len(rows)} conversations: {output}")
    print(f"Conditions: {dict(counts)}; longest prompt: {max(lengths)} tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
