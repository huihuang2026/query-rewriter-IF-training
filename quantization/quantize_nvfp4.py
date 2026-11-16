#!/usr/bin/env python3
"""Calibrate and quantize a merged BF16 model to NVFP4 (FP4 weights/activations).

Uses calibration JSONL produced by sample_calibration.py: balanced system/user
prompts with and without user instructions. No assistant labels are required.
Install quantization/requirements.txt in a separate quantization environment.
"""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

from quantize_w4a16 import MARKER, MODEL_NAME, ROOT, sha256


def load_calibration(path, expected_samples=None):
    samples = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid calibration JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Calibration line {line_number} must be an object")
            mode = row.get("user_instruction_mode")
            if mode not in ("with_user_instruction", "without_user_instruction"):
                raise ValueError(f"Missing or invalid user_instruction_mode on line {line_number}")
            messages = row.get("messages")
            if (not isinstance(messages, list) or len(messages) != 2
                    or any(not isinstance(m, dict) for m in messages)
                    or [m.get("role") for m in messages] != ["system", "user"]
                    or any(not isinstance(m.get("content"), str) or not m["content"].strip()
                           for m in messages)):
                raise ValueError(f"Line {line_number} needs nonempty system and user messages")
            has_instruction = bool(messages[0]["content"].partition(MARKER)[2].strip())
            if has_instruction != (mode == "with_user_instruction"):
                raise ValueError(f"Instruction condition differs from system prompt on line {line_number}")
            samples.append(row)
    if not samples:
        raise ValueError("Calibration file is empty")
    counts = dict(Counter(row["user_instruction_mode"] for row in samples))
    if (len(counts) != 2
            or counts["with_user_instruction"] != counts["without_user_instruction"]):
        raise ValueError(f"Expected equally many prompts in both instruction conditions: {counts}")
    if expected_samples is not None and len(samples) != expected_samples:
        raise ValueError(f"Expected {expected_samples} prompts; found {len(samples)}")
    manifest_path = path.with_suffix(".manifest.json")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    digest = sha256(path)
    if metadata.get("calibration_sha256", digest) != digest:
        raise ValueError("Calibration data differs from its sampling manifest; rerun sampling")
    metadata.update(calibration_file=str(path), calibration_sha256=digest,
                    num_samples=len(samples), condition_counts=counts)
    return samples, metadata


def quantize(args, samples, metadata):
    # Keep --help and --validate-only available without the quantization packages.
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("NVFP4 calibration requires a CUDA GPU")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized, truncated = [], 0
    for row in samples:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer(prompt, add_special_tokens=False)
        if len(tokens["input_ids"]) > args.max_seq_length:
            truncated += 1
            tokens = tokenizer(prompt, add_special_tokens=False, truncation=True,
                               max_length=args.max_seq_length)
        tokenized.append(dict(tokens))
    dataset = Dataset.from_list(tokenized)
    print(f"Tokenized {len(dataset)} prompts ({truncated} truncated to {args.max_seq_length} tokens)",
          flush=True)
    print(f"Loading merged BF16 checkpoint: {args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    model.eval()
    model.config.use_cache = False
    # Standard LLM Compressor NVFP4 W4A4 recipe: FP4 groups of 16 with
    # FP8 local scales and calibrated global scales. lm_head stays BF16;
    # KV-cache quantization is not enabled.
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
    print(f"Calibrating NVFP4 on all {len(samples)} prompts", flush=True)
    oneshot(model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe,
            max_seq_length=args.max_seq_length, num_calibration_samples=len(samples),
            shuffle_calibration_samples=False, batch_size=1)
    model.config.use_cache = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(args.output_dir))
    manifest = {
        "model_path": str(args.model_path), "source_config_sha256": sha256(args.model_path / "config.json"),
        "output_dir": str(args.output_dir), "scheme": "NVFP4", "algorithm": "PTQ",
        "weights": "FP4", "input_activations": "FP4", "group_size": 16,
        "ignore": ["lm_head"], "kv_cache_scheme": None, "seed": args.seed,
        "max_seq_length": args.max_seq_length, "truncated_prompts": truncated,
        "add_generation_prompt": True, "enable_thinking": False,
        "calibration": metadata,
        "versions": {package: version(package) for package in (
            "llmcompressor", "compressed-tensors", "transformers", "torch",
        )},
    }
    (args.output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved NVFP4 model: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "cpkts" / f"{MODEL_NAME}-BF16",
                        help="Local merged, unquantized checkpoint; do not pass FP8/INT4 weights")
    parser.add_argument("--calibration-file", type=Path, required=True,
                        help="Training calibration JSONL from sample_calibration.py")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: sibling of input checkpoint, replacing -BF16 with -NVFP4")
    parser.add_argument("--num-samples", type=int,
                        help="Optional expected prompt count; all rows are used, without resampling")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Calibration token cap; does not change inference context length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true",
                        help="Check paths, JSONL, balance, and sampling checksum without loading a model")
    args = parser.parse_args()
    if args.max_seq_length <= 0 or (args.num_samples is not None
                                   and (args.num_samples <= 0 or args.num_samples % 2)):
        parser.error("--max-seq-length must be positive; --num-samples must be positive and even")
    args.model_path = args.model_path.expanduser().resolve()
    args.calibration_file = args.calibration_file.expanduser().resolve()
    name = args.model_path.name.removesuffix("-BF16") + "-NVFP4"
    args.output_dir = (args.output_dir or args.model_path.with_name(name)).expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new --output-dir: {args.output_dir}")
    for path in (args.model_path / "config.json", args.calibration_file):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    try:
        config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
        if config.get("quantization_config"):
            raise ValueError("Input is already quantized; use the original merged BF16 checkpoint")
        samples, metadata = load_calibration(args.calibration_file, args.num_samples)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Model: {args.model_path}\nCalibration: {len(samples)} prompts\n"
          f"Output: {args.output_dir}", flush=True)
    if args.validate_only:
        print("Input validation passed; no model was loaded or quantized")
        return
    quantize(args, samples, metadata)


if __name__ == "__main__":
    main()
