#!/usr/bin/env python3
"""Merge the trained BF16 LoRA and save a vLLM-compatible FP8_DYNAMIC model.

Uses llmcompressor's data-free QuantizationModifier: static per-channel FP8
weights and dynamic per-token FP8 activations. No calibration file is required.
"""

import argparse
import hashlib
from importlib.metadata import version
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


def quantize(args):
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Run this quantization recipe in the CUDA quantization environment")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading BF16 base: {args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), dtype=torch.bfloat16, device_map="cpu", local_files_only=True,
    )
    print(f"Merging trained LoRA: {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(
        model, str(args.adapter), local_files_only=True, autocast_adapter_dtype=False,
    )
    model = model.merge_and_unload(safe_merge=True)
    model.eval()
    model.config.use_cache = False

    # Round-to-nearest weights; activation scales are calculated dynamically.
    # Embeddings and lm_head remain BF16. KV cache quantization is not enabled.
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    print("Applying FP8_DYNAMIC (W8A8); no calibration dataset required", flush=True)
    oneshot(model=model, tokenizer=tokenizer, recipe=recipe)

    model.config.use_cache = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(args.output_dir), save_compressed=True)
    tokenizer.save_pretrained(str(args.output_dir))
    manifest = {
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "output_dir": str(args.output_dir),
        "scheme": "FP8_DYNAMIC", "algorithm": "RTN", "seed": args.seed,
        "weights": {"type": "float", "num_bits": 8, "strategy": "channel", "dynamic": False},
        "input_activations": {"type": "float", "num_bits": 8, "strategy": "token", "dynamic": True},
        "ignore": ["lm_head"], "kv_cache_scheme": None,
        "calibration": None,
        "tokenizer": str(args.base_model),
        "versions": {package: version(package) for package in (
            "llmcompressor", "compressed-tensors", "transformers", "peft", "torch",
        )},
    }
    (args.output_dir / "quantization_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved FP8 model: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=ROOT / "models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path, default=ROOT / "ckpts" / MODEL_NAME / "final")
    parser.add_argument("--output-dir", type=Path, help="Default: sibling of adapter run, suffix -FP8-DYNAMIC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.base_model = args.base_model.expanduser().resolve()
    args.adapter = args.adapter.expanduser().resolve()
    run_dir = args.adapter.parent if args.adapter.name == "final" else args.adapter
    args.output_dir = (args.output_dir or run_dir.with_name(run_dir.name + "-FP8-DYNAMIC")).expanduser().resolve()
    if args.output_dir.exists():
        parser.error(f"Output already exists; choose a new --output-dir: {args.output_dir}")
    for path in (args.base_model / "config.json", args.adapter / "adapter_config.json",
                 args.adapter / "adapter_model.safetensors"):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    quantize(args)


if __name__ == "__main__":
    main()
