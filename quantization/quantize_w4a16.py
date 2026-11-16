#!/usr/bin/env python3
"""Step 2: merge the final LoRA and run W4A16 GPTQ on saved calibration prompts.

Run sample_calibration.py first. This script does not sample or change the data.
Dependencies: torch transformers peft datasets llmcompressor accelerate
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = (
    "Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-"
    "source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
)
MARKER = "Additional instruction from the user:"



def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantize(args, samples, metadata):
    # Lazy imports let --help work without importing CUDA/ML packages.
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("GPTQ requires a CUDA GPU")
    set_seed(args.seed)
    # Evaluation gets its tokenizer from the base vLLM model when using LoRA.
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized, truncated = [], 0
    for row in samples:
        # Keep these options identical to scripts/src/test_query_rewriter.py.
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
    print(f"Loading BF16 base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    print(f"Merging trained LoRA: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter), local_files_only=True,
                                     autocast_adapter_dtype=False)
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    model.config.use_cache = False
    # Same recipe as the previous deployment: INT4 weights, group size 128.
    # Activations, embeddings, and lm_head remain BF16.
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
    print(f"Running GPTQ on all {len(samples)} calibration prompts", flush=True)
    oneshot(model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe,
            max_seq_length=args.max_seq_length, num_calibration_samples=len(samples),
            shuffle_calibration_samples=False, batch_size=1)
    model.config.use_cache = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(args.output_dir))
    manifest = {
        "base_model": str(args.base_model), "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "output_dir": str(args.output_dir), "scheme": "W4A16", "algorithm": "GPTQ",
        "group_size": 128, "ignore": ["lm_head"], "max_seq_length": args.max_seq_length,
        "tokenizer": str(args.base_model), "add_generation_prompt": True, "enable_thinking": False,
        "calibration": metadata, "truncated_prompts": truncated,
    }
    (args.output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved quantized model: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-file", type=Path,
                        default=ROOT / "data/quantization/calibration_1024_seed42.jsonl")
    parser.add_argument("--adapter", type=Path, default=ROOT / "ckpts" / MODEL_NAME / "final")
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, help="Default: sibling of adapter run, suffix -W4A16-GPTQ")
    parser.add_argument("--num-samples", type=int, default=1024, help="Expected number of saved prompts")
    parser.add_argument("--max-seq-length", type=int, default=4096,
                        help="Calibration token limit; does not set the model's inference context limit")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.num_samples % 2 or args.max_seq_length <= 0:
        parser.error("--num-samples must be positive and even; --max-seq-length must be positive")
    for key in ("calibration_file", "adapter", "base_model"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    run_dir = args.adapter.parent if args.adapter.name == "final" else args.adapter
    args.output_dir = (args.output_dir or run_dir.with_name(run_dir.name + "-W4A16-GPTQ")).expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new --output-dir: {args.output_dir}")
    for path in (args.calibration_file, args.base_model / "config.json",
                 args.adapter / "adapter_config.json", args.adapter / "adapter_model.safetensors"):
        if not path.is_file():
            parser.error(f"Missing file: {path}; run sample_calibration.py first if calibration is missing")
    with args.calibration_file.open(encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    expected = {"with_user_instruction": args.num_samples // 2,
                "without_user_instruction": args.num_samples // 2}
    counts = dict(Counter(row["user_instruction_mode"] for row in samples))
    if len(samples) != args.num_samples or counts != expected:
        parser.error(f"Expected {args.num_samples} balanced prompts; found {len(samples)}: {counts}")
    for row in samples:
        messages = row.get("messages", [])
        if (len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]
                or any(not isinstance(m.get("content"), str) or not m["content"].strip()
                       for m in messages)):
            parser.error("Each calibration row must have nonempty system and user messages")
        has_instruction = bool(messages[0]["content"].partition(MARKER)[2].strip())
        if has_instruction != (row["user_instruction_mode"] == "with_user_instruction"):
            parser.error("Calibration condition does not match the system prompt")
    metadata_path = args.calibration_file.with_suffix(".manifest.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    digest = sha256(args.calibration_file)
    if metadata.get("calibration_sha256", digest) != digest:
        parser.error("Calibration data differs from its sampling manifest; rerun sampling")
    metadata.update(calibration_file=str(args.calibration_file), calibration_sha256=digest,
                    num_samples=len(samples), condition_counts=counts)
    print(f"Adapter: {args.adapter}\nCalibration: {args.calibration_file}\n"
          f"Quantized output: {args.output_dir}", flush=True)
    quantize(args, samples, metadata)


if __name__ == "__main__":
    main()
