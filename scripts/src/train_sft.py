#!/usr/bin/env python3
"""Unsloth query-rewriter SFT: LoRA or full tuning, three epochs, no validation.

Install Unsloth in a compatible CUDA/PyTorch environment, including its TRL,
Transformers, Datasets, and PEFT dependencies. See https://unsloth.ai/docs/get-started/install
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models/Qwen3-4B-Instruct-2507"
DEFAULT_INPUT = ROOT / "data/query_rewriter_self_instruct_summary_tokenized"


def tokenizer_fingerprint(tokenizer, *, ignore_post_processor=False):
    """Identify token IDs and chat formatting, independently of model weights."""
    from transformers import __version__ as transformers_version
    from tokenizers import __version__ as tokenizers_version

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        vocabulary = json.loads(backend.to_str())
        # Runtime padding/truncation is handled by the collator, not preprocessing.
        vocabulary.pop("padding", None)
        vocabulary.pop("truncation", None)
        if ignore_post_processor:
            # encode_example disables automatic special tokens for both chat
            # rendering and target encoding, so this processor is never used.
            vocabulary.pop("post_processor", None)
    else:
        vocabulary = tokenizer.get_vocab()
    value = {"vocabulary": vocabulary, "chat_template": tokenizer.chat_template,
             "special_tokens": tokenizer.special_tokens_map,
             "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id,
             "transformers_version": transformers_version, "tokenizers_version": tokenizers_version}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def validate_prepared_tokenizer(tokenizer, metadata, model_path):
    expected = metadata.get("tokenizer_fingerprint")
    if tokenizer_fingerprint(tokenizer) == expected:
        return
    from transformers import AutoTokenizer

    original = AutoTokenizer.from_pretrained(str(model_path))
    # First verify the exact tokenizer that actually produced the saved cache.
    # Keep the original fingerprint format so existing caches remain reusable.
    if tokenizer_fingerprint(original) != expected:
        raise ValueError("The model's source tokenizer differs from the prepared dataset; rerun preparation")
    if tokenizer_fingerprint(original, ignore_post_processor=True) != tokenizer_fingerprint(
            tokenizer, ignore_post_processor=True):
        raise ValueError("The training tokenizer changes tokenization or chat formatting; rerun preparation with this tokenizer")
    print("Tokenizer cache verified: Unsloth changed only the unused special-token post-processor")


def load_tokenized_dataset(input_path, max_length):
    from datasets import load_from_disk

    metadata_path = input_path / "preparation.json"
    if not metadata_path.is_file():
        raise ValueError(f"Tokenized dataset metadata missing: {metadata_path}; rerun preparation")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("max_seq_length") != max_length:
        raise ValueError("Tokenized dataset format or maximum length differs; rerun preparation")
    dataset = load_from_disk(str(input_path), keep_in_memory=False)
    if set(dataset.column_names) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Tokenized dataset must contain input_ids, attention_mask, and labels")
    if not len(dataset):
        raise ValueError("Tokenized dataset is empty")
    print(f"Loaded {len(dataset):,} pretokenized examples from {input_path}; tokenization skipped")
    return dataset, metadata


def encode_example(example, tokenizer, max_length):
    prompt, completion = example.get("prompt"), example.get("completion")
    if not isinstance(prompt, list) or [m.get("role") for m in prompt] != ["system", "user"]:
        raise ValueError("Expected system and user messages in prompt")
    if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
        raise ValueError("Expected one assistant completion")
    if not all(isinstance(m.get("content"), str) and m["content"].strip() for m in prompt + completion):
        raise ValueError("All messages must contain nonempty text")
    if tokenizer.eos_token_id is None:
        raise ValueError("The model tokenizer needs an EOS token")
    # Same generation prefix as inference. System-message text is never edited.
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True,
                   add_generation_prompt=True, enable_thinking=False, return_dict=False)
    target_ids = tokenizer.encode(completion[0]["content"], add_special_tokens=False)
    if not target_ids:
        raise ValueError("The query contains no target tokens")
    target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_length:
        return None
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids}


def tokenized_examples(input_path, tokenizer, max_length):
    total = kept = oversized = 0
    with Path(input_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            try:
                encoded = encode_example(json.loads(line), tokenizer, max_length)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"{input_path}:{line_number}: {error}") from error
            if encoded is None:
                oversized += 1
                continue
            kept += 1
            yield encoded
    print(f"Tokenized examples: {kept:,}/{total:,} | Oversized examples skipped: {oversized:,}")


def model_arguments(args, dtype):
    options = {"model_name": args.model, "max_seq_length": args.max_seq_length,
            "dtype": dtype, "load_in_4bit": args.load_in_4bit,
            "use_gradient_checkpointing": False, "offload_embedding": False,
            "full_finetuning": args.mode == "full"}
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # DDP replicates the model on each worker's GPU; never spread one
        # worker's model over the other workers' devices.
        options["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
    return options


def training_arguments(args, bf16):
    return {"output_dir": str(args.output_dir), "num_train_epochs": args.epochs,
            "max_steps": args.max_steps, "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate if args.learning_rate is not None else (2e-4 if args.mode == "lora" else 2e-5),
            "warmup_ratio": 0.03, "weight_decay": 0.01, "lr_scheduler_type": "linear",
            "optim": "adamw_torch", "bf16": bf16, "fp16": not bf16,
            "gradient_checkpointing": False, "logging_steps": 10,
            "ddp_find_unused_parameters": False,
            "eval_strategy": "no", "do_eval": False, "save_strategy": "epoch",
            "save_total_limit": 3, "load_best_model_at_end": False,
            "report_to": args.report_to, "run_name": args.run_name or args.output_dir.name,
            "seed": args.seed, "packing": False,
            "max_length": args.max_seq_length,
            # Explicit labels already mask all prompt tokens; do not mask twice.
            "completion_only_loss": False, "assistant_only_loss": False,
            "dataset_kwargs": {"skip_prepare_dataset": True},
            "remove_unused_columns": False}


def train(args):
    # Trainer's W&B callback initializes/logs only on the global main process.
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_ENTITY", "damian796-personal")
        os.environ.setdefault("WANDB_PROJECT", "query-rewriter-IF-training")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # Import Unsloth before Transformers/TRL so its supported optimizations apply.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    dataset = metadata = None
    if args.input.is_dir():
        # Open the memory-mapped dataset before allocating model weights on GPU.
        dataset, metadata = load_tokenized_dataset(args.input, args.max_seq_length)
    if not torch.cuda.is_available():
        raise ValueError("Training requires a supported CUDA GPU")
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(**model_arguments(args, dtype))
    if metadata is not None:
        validate_prepared_tokenizer(tokenizer, metadata, args.model)
    if args.mode == "lora":
        targets = args.lora_target_modules.split(",")
        available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        if any(name not in available for name in targets):
            raise ValueError(f"LoRA targets missing from model: {set(targets) - available}")
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=targets,
            # Favor speed: retain activations on GPU without checkpoint recomputation.
            lora_dropout=0, bias="none", use_gradient_checkpointing=False,
            random_state=args.seed)
    else:
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            raise ValueError("Full tuning requires an unquantized base model")
        model.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Mode: {args.mode} | Trainable parameters: {trainable:,}/{total:,}")
    if not trainable or (args.mode == "lora" and trainable == total) or (args.mode == "full" and trainable != total):
        raise ValueError("Trainable parameters do not match the requested training mode")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if dataset is None:
        # Keep existing JSON prompt/completion training scripts compatible.
        dataset = Dataset.from_generator(tokenized_examples, gen_kwargs={
            "input_path": str(args.input), "tokenizer": tokenizer,
            "max_length": args.max_seq_length})
    if not len(dataset):
        raise ValueError("No examples fit the requested sequence length")
    dataset = dataset.shuffle(seed=args.seed)
    if args.sample_size is not None:
        if args.sample_size > len(dataset):
            raise ValueError(f"Requested {args.sample_size:,} examples but only {len(dataset):,} are available")
        dataset = dataset.select(range(args.sample_size))
        print(f"Randomly sampled {len(dataset):,} training examples (seed={args.seed})")
    config = SFTConfig(**training_arguments(args, bf16))
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                        args=config, data_collator=DataCollatorForSeq2Seq(
                            tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
                            return_tensors="pt"))
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
    if args.merge_lora and trainer.is_world_process_zero():
        model.save_pretrained_merged(str(args.output_dir / "merged"), tokenizer,
                                    save_method="merged_16bit")
    print(f"Final {'LoRA adapter' if args.mode == 'lora' else 'model'}: {final_dir}")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Prompt/completion JSON Lines file or prepared tokenized Arrow directory")
    parser.add_argument("--output-dir", type=Path, help="Default: ckpts/<model-name>-LoRA for LoRA, or ckpts/<model-name> for full tuning")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use QLoRA; valid only with --mode lora")
    parser.add_argument("--epochs", type=positive, default=3)
    parser.add_argument("--batch-size", type=positive, default=1)
    parser.add_argument("--gradient-accumulation", type=positive, default=16)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-seq-length", type=positive, default=32768)
    parser.add_argument("--lora-rank", type=positive, default=16)
    parser.add_argument("--lora-alpha", type=positive, default=32)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=positive, help="Randomly sample this many training examples")
    parser.add_argument("--max-steps", type=positive, default=-1, help="Optional short training run instead of all epochs")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none",
                        help="Enable W&B training charts with --report-to wandb")
    parser.add_argument("--run-name", help="W&B display name; defaults to the output directory name")
    parser.add_argument("--merge-lora", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        suffix = "-LoRA" if args.mode == "lora" else ""
        args.output_dir = ROOT / "ckpts" / f"{Path(args.model).name}{suffix}"
    if args.mode == "full" and (args.load_in_4bit or args.merge_lora):
        parser.error("Full tuning cannot use --load-in-4bit or --merge-lora")
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than zero")
    if not args.input.is_file() and not args.input.is_dir():
        parser.error(f"Processed input file or tokenized dataset directory not found: {args.input}")
    return args


def main():
    args = parse_args()
    try:
        train(args)
    except ImportError as error:
        raise SystemExit(f"Missing training dependency: {error}. Install Unsloth and its compatible TRL/Datasets dependencies.") from error


if __name__ == "__main__":
    main()
