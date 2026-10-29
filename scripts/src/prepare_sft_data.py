#!/usr/bin/env python3
"""Build self-instruct SFT examples with optional score filtering and tokenization."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import tempfile
import unicodedata

from train_sft import DEFAULT_MODEL, encode_example, tokenized_examples, tokenizer_fingerprint

ROOT = Path(__file__).resolve().parents[2]
MARKER = "Additional instruction from the user:"


def records(path):
    # Only read complete lines from the snapshot, even if construction is ongoing.
    with path.open("rb") as handle:
        size = path.stat().st_size
        while handle.tell() < size:
            line = handle.readline()
            if handle.tell() > size or not line.endswith(b"\n"):
                break
            if line.strip():
                yield json.loads(line)


def normalized(text):
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def valid_query(value):
    return isinstance(value, str) and value.strip() not in ("", "[NO_QUERY]")


def example(system, conversation, query):
    return {"prompt": [{"role": "system", "content": system},
                       {"role": "user", "content": conversation}],
            "completion": [{"role": "assistant", "content": query}]}


def save(path, rows):
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


def passes_score_thresholds(row, quality_min, following_min):
    """Apply enabled thresholds to the two responses in the same record."""
    if quality_min is None and following_min is None:
        return True
    if row.get("scoring_error"):
        return False
    scores = row.get("response_scores") or {}
    for metric, minimum in (("query_quality", quality_min),
                            ("instruction_following", following_min)):
        if minimum is not None:
            score = (scores.get(metric) or {}).get("score")
            if type(score) is not int or not minimum <= score <= 5:
                return False
    return True


def eligible_conversations(args):
    """Collect valid, unique instruction pairs while excluding held-out data."""
    blocked_text, blocked_instruction = set(), set()
    for row in records(args.test):
        source = row.get("source", row)
        blocked_text.add(normalized(source["user_content"]))
        instruction = source["system_prompt"].partition(MARKER)[2]
        if not instruction.strip():
            raise ValueError("Test record has no additional instruction")
        blocked_instruction.add(normalized(instruction))
    groups = defaultdict(dict)
    total = score_passed = 0
    for path in args.input:
        for row in records(path):
            total += 1
            if not passes_score_thresholds(row, args.min_query_quality, args.min_instruction_following):
                continue
            score_passed += 1
            conversation = row.get("user_content")
            base = row.get("system_prompt_without_user_instruction")
            instruction = row.get("user_defined_instruction")
            if not all(isinstance(v, str) and v.strip() for v in (conversation, base, instruction)):
                continue
            if row.get("generation_error") or row.get("generation_skipped") or not all(valid_query(row.get(k)) for k in
                    ("query_with_user_instruction", "query_without_user_instruction")):
                continue
            if normalized(conversation) in blocked_text or normalized(instruction) in blocked_instruction:
                continue
            if row.get("system_prompt_with_user_instruction") != base + MARKER + "\n" + instruction:
                raise ValueError("Stored full prompt does not match its base prompt and additional instruction")
            groups[normalized(conversation)].setdefault(normalized(instruction), row)
    if args.min_query_quality is not None or args.min_instruction_following is not None:
        print(f"Score filter: {score_passed:,}/{total:,} pairs passed "
              f"(query quality >= {args.min_query_quality}, "
              f"instruction following >= {args.min_instruction_following})", flush=True)
        if not score_passed:
            raise ValueError("No pairs meet the requested score thresholds")
    eligible = {key: list(group.values()) for key, group in groups.items()
                if len(group) >= args.instruction_count}
    return eligible


def sample_pairs(args):
    """Randomly sample one constructed pair per conversation, retaining provenance."""
    if args.instruction_count != 1:
        raise ValueError("--sample-pairs selects one instruction pair per conversation; use --instruction-count 1")
    if args.output is None:
        raise ValueError("--sample-pairs requires an explicit --output")
    if args.output.resolve() in {p.resolve() for p in args.input} | {args.test.resolve()}:
        raise ValueError("Sample output must differ from source files")
    eligible = eligible_conversations(args)
    count = len(eligible) if args.examples is None else args.examples
    if count < 1:
        raise ValueError("No completed, nonempty instruction pairs remain after exclusions")
    if count > len(eligible):
        raise ValueError(f"Requested {count:,} conversation pairs but only {len(eligible):,} are eligible; "
                         "sample output unchanged")
    rng = random.Random(args.seed)
    selected = rng.sample(sorted(eligible), count)
    rows = [rng.choice(eligible[key]) for key in selected]
    save(args.output, rows)
    score_filter = (f"query quality >= {args.min_query_quality}, instruction following >= {args.min_instruction_following}"
                    if args.min_query_quality is not None or args.min_instruction_following is not None
                    else "no score filtering")
    print(f"Randomly sampled {count:,}/{len(eligible):,} conversation pairs (seed={args.seed}); "
          f"{score_filter}. Saved: {args.output}", flush=True)
    return args.output


def build(args):
    eligible = eligible_conversations(args)
    per_conversation = args.instruction_count + 1
    if args.examples is None:
        needed = len(eligible)
        if not needed:
            raise ValueError("No conversations have enough completed, nonempty response pairs")
    else:
        if args.examples % per_conversation:
            raise ValueError(f"--examples must be divisible by {per_conversation}")
        needed = args.examples // per_conversation
    print(f"Eligible conversations: {len(eligible):,}; required: {needed:,}")
    if len(eligible) < needed:
        raise ValueError(f"Need {needed - len(eligible):,} more conversations with completed responses; "
                         "existing conversations will not be duplicated to inflate dataset size")
    rng = random.Random(args.seed)
    selected = rng.sample(sorted(eligible), needed)
    output = []
    for key in selected:
        variants = rng.sample(eligible[key], args.instruction_count)
        first = variants[0]
        output.append(example(first["system_prompt_without_user_instruction"], first["user_content"],
                              first["query_without_user_instruction"]))
        for row in variants:
            if row["system_prompt_without_user_instruction"] != first["system_prompt_without_user_instruction"]:
                raise ValueError("Variants of one conversation have different base prompts")
            output.append(example(row["system_prompt_with_user_instruction"], row["user_content"],
                                  row["query_with_user_instruction"]))
    assert len(output) == needed * per_conversation
    rng.shuffle(output)
    path = getattr(args, "output", None) or args.output_dir / f"query_writer_self_instruct_{args.instruction_count}_{len(output)}.jsonl"
    if path.resolve() in {p.resolve() for p in args.input} | {args.test.resolve()}:
        raise ValueError("Output must differ from source files")
    save(path, output)
    print(f"Saved {len(output):,} self-instruct-only examples from {needed:,} conversations: {path}")
    return path


def example_count(value):
    if value.strip().lower() in ("null", "none", "all"):
        return None
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--examples must be positive or NULL")
    return count


def file_signature(path):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while checking the cache: {path}; rerun after construction finishes")
    return {"path": str(path.resolve()), "sha256": checksum.hexdigest(),
            "size": after.st_size, "mtime_ns": after.st_mtime_ns}


def build_tokenized(args):
    from datasets import Dataset, Features, Sequence, Value
    from transformers import AutoTokenizer

    target = args.tokenized_output.resolve()
    protected = list(args.input) + [args.test, Path(args.model)]
    if args.output is not None:
        protected.append(args.output)
    if any(target == path.resolve() or target in path.resolve().parents for path in protected):
        raise ValueError("Tokenized output directory must not contain or replace any input/model/JSON output")
    if target.exists() and (not target.is_dir() or (
            any(target.iterdir()) and not (target / "preparation.json").is_file())):
        raise ValueError("Tokenized output already exists and is not a prepared dataset; choose another directory")
    print("Checking tokenized dataset cache...", flush=True)
    sources = [file_signature(path) for path in [*args.input, args.test]]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    tokenizer_id = tokenizer_fingerprint(tokenizer)
    algorithm = "\n".join(inspect.getsource(function) for function in (
        build, eligible_conversations, passes_score_thresholds, records, normalized, valid_query, example,
        encode_example, tokenized_examples))
    identity = {"format_version": 1, "sources": [
        {key: value for key, value in source.items() if key != "mtime_ns"} for source in sources],
        "instruction_count": args.instruction_count, "examples": args.examples,
        "min_query_quality": args.min_query_quality,
        "min_instruction_following": args.min_instruction_following,
        "seed": args.seed, "max_seq_length": args.max_seq_length,
        "tokenizer_fingerprint": tokenizer_id,
        "preprocessing_code": hashlib.sha256(algorithm.encode()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = target / "preparation.json"
    if manifest.is_file() and not args.rebuild_cache:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if metadata.get("cache_fingerprint") == fingerprint and (target / "state.json").is_file():
            print(f"Reusing {metadata['num_examples']:,} tokenized examples: {target}. "
                  "Selection and tokenization skipped.", flush=True)
            return target

    processed = build(args)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        temporary = Path(temporary)
        features = Features({"input_ids": Sequence(Value("int32")),
                             "attention_mask": Sequence(Value("int8")),
                             "labels": Sequence(Value("int32"))})
        dataset = Dataset.from_generator(tokenized_examples,
            gen_kwargs={"input_path": str(processed), "tokenizer": tokenizer,
                        "max_length": args.max_seq_length},
            features=features, cache_dir=str(temporary / "build_cache"), fingerprint=fingerprint)
        if not len(dataset):
            raise ValueError("No examples fit the requested sequence length; the previous cache was preserved")
        staged = temporary / "dataset"
        dataset.save_to_disk(str(staged))
        metadata = {**identity, "cache_fingerprint": fingerprint,
                    "tokenizer_model": str(args.model), "num_examples": len(dataset)}
        (staged / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        for source in sources:
            current = Path(source["path"]).stat()
            if (current.st_size, current.st_mtime_ns) != (source["size"], source["mtime_ns"]):
                raise ValueError("Input changed during preparation; previous cache preserved. Rerun on finished data.")
        # Publish only a complete dataset, retaining the previous cache on failure.
        previous = temporary / "previous"
        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(staged, target)
        except BaseException:
            if previous.exists():
                os.replace(previous, target)
            raise
    print(f"Saved {metadata['num_examples']:,} tokenized examples: {target}", flush=True)
    return target


def score_threshold(value):
    threshold = int(value)
    if not 1 <= threshold <= 5:
        raise argparse.ArgumentTypeError("score thresholds must be between 1 and 5")
    return threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more completed Gemini response files")
    parser.add_argument("--instruction-count", type=int, required=True)
    parser.add_argument("--min-query-quality", type=score_threshold,
                        help="Minimum query-quality score; omit to disable this threshold")
    parser.add_argument("--min-instruction-following", type=score_threshold,
                        help="Minimum instruction-following score; both thresholds apply to the same pair")
    parser.add_argument("--examples", type=example_count, default=6000,
                        help="Final SFT examples, or conversation pairs with --sample-pairs; NULL uses all eligible conversations")
    parser.add_argument("--sample-pairs", action="store_true",
                        help="Save a random sample of constructed pairs before prompt/completion preparation; optional score thresholds apply first")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, help="Optional explicit output filename; content stays JSON Lines")
    parser.add_argument("--test", type=Path, default=ROOT / "data/query_rewriter_exported_sample10k_random1k_filtered.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenized-output", type=Path,
                        help="Optional Arrow dataset directory; cache hits skip preparation and tokenization")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Tokenizer directory; no model weights are loaded")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild even if the tokenized cache matches")
    args = parser.parse_args()
    if args.instruction_count < 1:
        parser.error("--instruction-count must be positive")
    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be positive")
    if args.sample_pairs and args.tokenized_output is not None:
        parser.error("Sample pairs first, then tokenize the sampled file in a separate command")
    try:
        if args.sample_pairs:
            sample_pairs(args)
        elif args.tokenized_output is not None:
            build_tokenized(args)
        else:
            build(args)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
