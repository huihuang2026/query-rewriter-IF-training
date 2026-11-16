#!/usr/bin/env python3
"""Merge the original train02 adapter into BF16 weights for the speed baseline."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = (
    "Qwen3-4B-final-train02-700K-SFT-previous-DPO-length-control-"
    "source20K-IF4-CP4-withinst-noinst-len1.0-sft1.0-1epoch"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading BF16 base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    print(f"Merging original train02 LoRA: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(
        model, str(args.adapter), local_files_only=True, autocast_adapter_dtype=False,
    )
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    model.config.use_cache = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    manifest = {
        "base_model": str(args.base_model), "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "output_dir": str(args.output_dir), "dtype": "bfloat16", "quantization": None,
        "method": "PEFT merge_and_unload(safe_merge=True)", "autocast_adapter_dtype": False,
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved original merged BF16 model: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, default=ROOT / "ckpts" / MODEL_NAME / "final")
    parser.add_argument("--output-dir", type=Path, help="Default: sibling of adapter run, suffix -BF16")
    args = parser.parse_args()
    args.base_model = args.base_model.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    run_dir = args.adapter.parent if args.adapter.name == "final" else args.adapter
    args.output_dir = (args.output_dir or run_dir.with_name(run_dir.name + "-BF16")).expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Output already exists: {args.output_dir}; comment out the merge step to serve it")
    for path in (args.base_model / "config.json", args.adapter / "adapter_config.json",
                 args.adapter / "adapter_model.safetensors"):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    merge(args)


if __name__ == "__main__":
    main()
