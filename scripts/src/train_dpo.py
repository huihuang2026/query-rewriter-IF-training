#!/usr/bin/env python3
r"""Continue the SFT LoRA adapter with DPO on Gemini-chosen / SFT-rejected pairs.

Run from the workspace root (training is never launched by construction scripts):
    .venv-finetune/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
        scripts/src/train_dpo.py --batch-size 1 --gradient-accumulation 4

Uses the installed Unsloth, TRL, Transformers, PEFT, and Datasets packages.
The frozen DPO reference is the initial SFT adapter, not the original Qwen model.
There is no validation split; checkpoints are saved each epoch and under final/.
"""
import argparse
import math
import os
from pathlib import Path

from prepare_sft_data import normalized, records, valid_query

ROOT = Path(__file__).resolve().parents[2]
SFT_NAME = "Qwen3-4B-Instruct-2507-self_instruct_summary_1_all-LoRA"
DEFAULT_INPUT = ROOT / "data/query_rewriter_self_instruct_dpo_train.json"
DEFAULT_ADAPTER = ROOT / "ckpts" / SFT_NAME / "final"


def validate_pair(row):
    """Accept only explicit conversational preferences, preserving their text."""
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    for field, roles in [("prompt", ["system", "user"]),
                         ("chosen", ["assistant"]), ("rejected", ["assistant"])]:
        messages = row.get(field)
        if (not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages)
                or [m.get("role") for m in messages] != roles):
            raise ValueError(f"{field} must contain messages with roles {roles}")
        if not all(valid_query(m.get("content")) for m in messages):
            raise ValueError(f"{field} contains an empty or [NO_QUERY] message")
    return {field: row[field] for field in ("prompt", "chosen", "rejected")}


def preference_examples(input_path, tokenizer, max_length, input_signature=None, strict_pairs=False):
    """Filter whole pairs before TRL tokenization; never truncate either branch."""
    # input_signature participates in the Datasets cache key, so a newly prepared
    # file at the same path cannot silently reuse an older training dataset.
    if input_signature is not None:
        stat = Path(input_path).stat()
        if (stat.st_size, stat.st_mtime_ns) != tuple(input_signature):
            raise ValueError("DPO input changed while preparing training; rerun with the finished file")
    total = kept = identical = oversized = 0
    for total, row in enumerate(records(Path(input_path)), 1):
        try:
            pair = validate_pair(row)
            if normalized(pair["chosen"][0]["content"]) == normalized(pair["rejected"][0]["content"]):
                identical += 1
                if not strict_pairs:
                    continue
            prompt_ids = tokenizer.apply_chat_template(
                pair["prompt"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=False)
            full_ids = [tokenizer.apply_chat_template(
                pair["prompt"] + pair[field], tokenize=True, enable_thinking=False,
                return_dict=False) for field in ("chosen", "rejected")]
            # A different prefix would train on a context that differs from inference.
            if any(ids[:len(prompt_ids)] != prompt_ids for ids in full_ids):
                raise ValueError("Chat template changes the prompt prefix when an answer is appended")
            if any(len(ids) > max_length for ids in full_ids):
                if strict_pairs:
                    raise ValueError("Pair exceeds max length; refusing to change the controlled sample")
                oversized += 1
                continue
            kept += 1
            yield {**pair, "chat_template_kwargs": {"enable_thinking": False}}
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError(f"{input_path}: record {total}: {error}") from error
    print(f"DPO pairs: {kept:,}/{total:,} | Identical {'retained' if strict_pairs else 'skipped'}: {identical:,} | "
          f"Overlength skipped: {oversized:,}", flush=True)
    if not kept:
        raise ValueError("No usable DPO pairs fit the requested sequence length")


def verify_sft_reference(model):
    """Require TRL's frozen SFT copy; never silently use disabled-adapter Qwen."""
    import torch

    if not {"default", "ref"}.issubset(model.peft_config):
        raise ValueError("TRL did not create a reference copy of the SFT adapter. "
                         "Use the compatible TRL 1.15+ / PEFT environment.")
    model.set_adapter("default")
    references = 0
    for name, parameter in model.named_parameters():
        if ".ref." in name:
            references += 1
            original = model.get_parameter(name.replace(".ref.", ".default."))
            if parameter.requires_grad or not torch.equal(parameter.detach(), original.detach()):
                raise ValueError("The DPO reference must be a frozen, exact copy of the SFT adapter")
        elif parameter.requires_grad and ".default." not in name:
            raise ValueError(f"Unexpected trainable parameter outside the SFT adapter: {name}")
    if not references or not any(p.requires_grad for p in model.parameters()):
        raise ValueError("Missing frozen reference weights or trainable SFT adapter weights")


def config_arguments(args, bf16, num_pairs):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_steps = math.ceil(math.ceil(num_pairs / world_size) / args.batch_size)
    total_steps = (args.max_steps if args.max_steps > 0 else
                   math.ceil(micro_steps / args.gradient_accumulation) * args.epochs)
    return dict(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, beta=args.beta,
        loss_type=["sigmoid", "sft"] if args.sft_loss_weight > 0 else "sigmoid",
        loss_weights=[1.0, args.sft_loss_weight] if args.sft_loss_weight > 0 else None,
        warmup_steps=math.ceil(total_steps * 0.03), lr_scheduler_type="linear", weight_decay=0.01,
        optim="adamw_torch", bf16=bf16, fp16=not bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False, eval_strategy="no", do_eval=False,
        save_strategy="epoch", save_total_limit=3, logging_steps=10, logging_first_step=True,
        report_to=args.report_to, run_name=args.run_name or args.output_dir.name,
        seed=args.seed, max_length=args.max_seq_length,
        # Both complete branches were checked above; the collator will not truncate them.
        truncation_mode="keep_start", disable_dropout=True,
        precompute_ref_log_probs=False, activation_offloading=False,
    )


def train(args):
    # Trainer's W&B callback initializes/logs only on the global main process.
    if args.report_to == "wandb":
        os.environ.setdefault("WANDB_ENTITY", "damian796-personal")
        os.environ.setdefault("WANDB_PROJECT", "query-rewriter-IF-training")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("DPO training requires CUDA GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # Select this worker's GPU before Unsloth initializes; import Unsloth before
    # TRL and Transformers so its optimizations apply.
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    PatchDPOTrainer()
    bf16 = torch.cuda.is_bf16_supported()
    # Loading the existing adapter restores the SFT weights and its LoRA settings.
    # Do not call get_peft_model: that would initialize a new adapter.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.sft_adapter), max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if bf16 else torch.float16, load_in_4bit=False,
        full_finetuning=False, use_gradient_checkpointing=args.gradient_checkpointing,
        offload_embedding=False, device_map={"": local_rank},
    )
    if not hasattr(model, "peft_config") or set(model.peft_config) != {"default"}:
        raise ValueError("Expected the single trained SFT LoRA adapter named default")
    FastLanguageModel.for_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model.config.use_cache = False
    if tokenizer.eos_token_id is None:
        raise ValueError("The SFT tokenizer needs an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    stat = args.input.stat()
    dataset = Dataset.from_generator(preference_examples, gen_kwargs={
        "input_path": str(args.input), "tokenizer": tokenizer, "max_length": args.max_seq_length,
        "input_signature": (stat.st_size, stat.st_mtime_ns),
        "strict_pairs": args.strict_pairs,
    }).shuffle(seed=args.seed)
    config = DPOConfig(**config_arguments(args, bf16, len(dataset)))
    # For an already trained PEFT model, current TRL clones default -> ref and
    # switches to that frozen adapter for reference log probabilities.
    # No peft_config is passed: preserve and continue the existing SFT adapter.
    trainer = DPOTrainer(model=model, ref_model=None, args=config,
                         processing_class=tokenizer, train_dataset=dataset)
    verify_sft_reference(model)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if trainer.is_world_process_zero():
        print(f"SFT initialization and frozen reference: {args.sft_adapter}", flush=True)
        print(f"Loss: DPO + {args.sft_loss_weight:g} * chosen-answer SFT", flush=True)
        print(f"DPO pairs: {len(dataset):,} | GPUs: {world_size} | Effective batch: "
              f"{args.batch_size * args.gradient_accumulation * world_size} preference pairs", flush=True)
        print("Each preference pair contains one chosen and one rejected answer.", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Trainer checkpoints retain both adapters for resume. The deployable final
    # artifact contains only the trained policy adapter, based on the same Qwen.
    trainer.save_state()
    if trainer.is_world_process_zero():
        final_dir = args.output_dir / "final"
        model.save_pretrained(str(final_dir), selected_adapters=["default"])
        tokenizer.save_pretrained(str(final_dir))
        print(f"Final DPO LoRA adapter: {final_dir}", flush=True)
    trainer.accelerator.wait_for_everyone()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Prepared DPO prompt/chosen/rejected JSON Lines file")
    parser.add_argument("--sft-adapter", type=Path, default=DEFAULT_ADAPTER,
                        help="Finished SFT LoRA adapter: initializes both policy and frozen reference")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: ckpts/<SFT model name>-DPO")
    parser.add_argument("--epochs", type=positive, default=1)
    parser.add_argument("--batch-size", type=positive, default=1, help="Preference pairs per GPU")
    parser.add_argument("--gradient-accumulation", type=positive, default=4)
    parser.add_argument("--learning-rate", type=positive_float, default=5e-6)
    parser.add_argument("--beta", type=positive_float, default=0.1, help="DPO reference regularization strength")
    parser.add_argument("--sft-loss-weight", type=nonnegative_float, default=0.0,
                        help="Chosen-answer SFT loss weight added to DPO; 0 keeps DPO only")
    parser.add_argument("--strict-pairs", action="store_true",
                        help="Keep identical pairs and fail on overlength pairs to preserve a controlled sample")
    parser.add_argument("--max-seq-length", type=positive, default=4096)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Optional GPU activation recomputation; disabled by default for speed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=positive, default=-1, help="Override epochs for a short run")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--report-to", choices=("none", "wandb"), default="none",
                        help="Enable W&B training charts with --report-to wandb")
    parser.add_argument("--run-name", help="W&B display name; defaults to the output directory name")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"DPO training file not found: {args.input}")
    if not (args.sft_adapter / "adapter_config.json").is_file():
        parser.error(f"SFT adapter_config.json not found in {args.sft_adapter}")
    if not any((args.sft_adapter / name).is_file()
               for name in ("adapter_model.safetensors", "adapter_model.bin")):
        parser.error(f"SFT adapter weights not found in {args.sft_adapter}")
    if args.output_dir is None:
        model_name = args.sft_adapter.parent.name if args.sft_adapter.name == "final" else args.sft_adapter.name
        args.output_dir = ROOT / "ckpts" / f"{model_name}-DPO"
    output, sft = args.output_dir.resolve(), args.sft_adapter.resolve()
    if output.is_relative_to(sft) or sft.is_relative_to(output):
        parser.error("Use a separate output directory to preserve the SFT checkpoint")
    if args.resume_from_checkpoint and not args.resume_from_checkpoint.is_dir():
        parser.error(f"Resume checkpoint not found: {args.resume_from_checkpoint}")
    if not args.resume_from_checkpoint and (
            (output / "final").exists() or any(output.glob("checkpoint-*"))):
        parser.error("Output already contains a training run; use --resume-from-checkpoint or a new output directory")
    return args


def main():
    args = parse_args()
    try:
        train(args)
    except ImportError as error:
        raise SystemExit(f"Missing training dependency: {error}. Use .venv-finetune with compatible "
                         "Unsloth / TRL 1.15+ / PEFT packages.") from error


if __name__ == "__main__":
    main()
