#!/usr/bin/env python3
"""Freeze shared prompts/positives, then vary only the model supplying negatives."""
import argparse
import fcntl
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

import construct_dpo_data as dpo
from train_dpo import validate_pair


def load(path):
    rows = list(dpo.records(path))
    if len({r['record_id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate record IDs: {path}')
    return rows


def prompt(row):
    return [{'role': 'system', 'content': row['source']['system_prompt']},
            {'role': 'user', 'content': row['source']['user_content']}]


def sample(args):
    settings = dict(source=str(args.input.resolve()), test=str(args.test.resolve()),
                    sample_size=args.size, applicability_min=args.applicability,
                    seed=args.seed, model=str(args.model.resolve()),
                    negative_model=args.negative_model, max_length=args.max_length,
                    max_new_tokens=args.max_new_tokens)
    if args.output.exists():
        rows = load(args.output)
        if len(rows) != args.size or any(r.get('sample_settings') != settings for r in rows):
            raise ValueError('Shared sample settings differ; use a new sample path for a new experiment')
        # The shared generation entry point validates this metadata against the prompt.
        if any('user_defined_instruction' not in r for r in rows):
            for r in rows:
                r['user_defined_instruction'] = r['source']['system_prompt'].partition(dpo.MARKER)[2].strip()
            dpo.save(args.output, rows)
        print(f'Reusing fixed {len(rows):,} prompts/positives: {args.output}')
        return
    blocked_conversations, blocked_instructions = dpo.held_out(args.test)
    counts = Counter()
    # Last occurrence per ID wins, without loading the entire source into RAM.
    with args.input.open('rb') as f:
        offsets = {}
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                r = json.loads(line)
                offsets[r['record_id']] = offset
        eligible = []
        for offset in offsets.values():
            f.seek(offset)
            r = json.loads(f.readline())
            if dpo.instruction_mode(r) != dpo.WITH_INSTRUCTION:
                counts['without_instruction'] += 1
                continue
            score = r.get('instruction_evaluation', {}).get('sft', {}).get('instruction_applicability', {}).get('score')
            if type(score) is not int or not args.applicability <= score <= 5:
                counts['applicability'] += 1
                continue
            if r.get('generation_error') or not all(dpo.valid_query(r.get(k)) for k in ('gemini_query', 'generated_query')):
                counts['invalid_or_missing_answers'] += 1
                continue
            if r.get('generation_model') != args.negative_model:
                raise ValueError('Unexpected cached negative model; do not mix generators')
            instruction = r['source']['system_prompt'].partition(dpo.MARKER)[2].strip()
            if not instruction:
                raise ValueError('Conditioned row has no additional instruction')
            if (dpo.conversation_key(r['source']['user_content']) in blocked_conversations
                    or dpo.normalized(instruction) in blocked_instructions):
                counts['test_overlap'] += 1
                continue
            config = r.get('generation_config', {})
            if config.get('temperature') != 0 or config.get('max_new_tokens') != args.max_new_tokens:
                raise ValueError('Cached decoding settings differ from the planned old-model generation')
            eligible.append(offset)
        print(f'Eligible before context checks: {len(eligible):,}; exclusions: {dict(counts)}', flush=True)
        random.Random(args.seed).shuffle(eligible)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        selected = []
        for offset in eligible:
            f.seek(offset)
            r = json.loads(f.readline())
            p = prompt(r)
            prompt_tokens = tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True,
                                                         enable_thinking=False, return_dict=False)
            full = [tokenizer.apply_chat_template(p + [{'role': 'assistant', 'content': r[k]}],
                       tokenize=True, enable_thinking=False, return_dict=False)
                    for k in ('gemini_query', 'generated_query')]
            # Reserve the same decoding budget for the old model, plus chat termination tokens.
            if len(prompt_tokens) + args.max_new_tokens + 8 > args.max_length or any(len(x) > args.max_length for x in full):
                counts['context_budget'] += 1
                continue
            keep = {k: r.get(k) for k in ('record_id', 'source_record_id', 'source', 'gemini_query',
                    'gemini_model', 'generated_query', 'generation_model', 'generation_config')}
            keep.update(user_instruction_mode=dpo.WITH_INSTRUCTION,
                        user_defined_instruction=r['source']['system_prompt'].partition(dpo.MARKER)[2].strip(),
                        instruction_applicability_score=r['instruction_evaluation']['sft']['instruction_applicability']['score'],
                        sample_settings=settings)
            selected.append(keep)
            if len(selected) == args.size:
                break
    if len(selected) != args.size:
        raise ValueError(f'Only {len(selected)} usable pairs; need {args.size}. No sample written.')
    dpo.save(args.output, selected)
    print(f'Saved {len(selected):,} shared pairs: {args.output}; exclusions: {dict(counts)}')
    print('No IF/quality, comparison-winner, or relative-length filtering. Identical answers retained.')


def sample_noinst(args):
    """Use valid cached base-only counterparts of the selected instructed conversations."""
    parents = load(args.conditioned_sample)
    settings = dict(source=str(args.input.resolve()), conditioned_sample=str(args.conditioned_sample.resolve()),
                    conditioned_hash=dpo.file_hash(args.conditioned_sample), negative_model=args.negative_model,
                    max_length=args.max_length, max_new_tokens=args.max_new_tokens)
    if args.output.exists():
        rows = load(args.output)
        parent_ids = {r['record_id'] for r in parents}
        if (not rows or len(rows) > len(parents)
                or any(r.get('sample_settings') != settings or r.get('conditioned_record_id') not in parent_ids for r in rows)):
            raise ValueError('No-instruction sample differs; use a new output path')
        print(f'Reusing fixed {len(rows):,} no-instruction pairs: {args.output}')
        return
    expected = {}
    for r in parents:
        system, marker, instruction = r['source']['system_prompt'].partition(dpo.MARKER)
        if not marker or not instruction.strip():
            raise ValueError('Expected an instructed parent sample')
        source = dict(system_prompt=system, user_content=r['source']['user_content'])
        key = dpo.digest([system, source['user_content']])
        if key in expected:
            raise ValueError('Multiple selected instructions share a base prompt; cannot preserve a unique 10K sample')
        expected[key] = (source, r['record_id'])
    found = {}
    for r in dpo.records(args.input):
        if r['record_id'] in expected:
            found[r['record_id']] = r
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    selected = []
    errors = Counter()
    for key, (source, parent_id) in expected.items():
        r = found.get(key)
        if r is None:
            errors['missing_base_counterpart'] += 1
            continue
        if r['source'] != source or dpo.instruction_mode(r) != dpo.WITHOUT_INSTRUCTION:
            raise ValueError('Base counterpart prompt differs from the instructed conversation')
        if r.get('generation_error') or not all(dpo.valid_query(r.get(k)) for k in ('gemini_query', 'generated_query')):
            errors['invalid_or_missing_answers'] += 1
            continue
        config = r.get('generation_config', {})
        if r.get('generation_model') != args.negative_model or config.get('temperature') != 0 or config.get('max_new_tokens') != args.max_new_tokens:
            raise ValueError('Base counterpart generator/decoding settings differ')
        p = prompt(r)
        ids = tokenizer.apply_chat_template(p, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_dict=False)
        full = [tokenizer.apply_chat_template(p + [{'role': 'assistant', 'content': r[k]}],
                tokenize=True, enable_thinking=False, return_dict=False)
                for k in ('gemini_query', 'generated_query')]
        if len(ids) + args.max_new_tokens + 8 > args.max_length or any(len(x) > args.max_length for x in full):
            errors['context_budget'] += 1
            continue
        keep = {k: r.get(k) for k in ('record_id', 'source_record_id', 'source', 'gemini_query',
                'gemini_model', 'generated_query', 'generation_model', 'generation_config')}
        keep.update(user_instruction_mode=dpo.WITHOUT_INSTRUCTION, user_defined_instruction='',
                    conditioned_record_id=parent_id, sample_settings=settings)
        selected.append(keep)
    if not selected:
        raise ValueError(f'No usable base counterparts: {dict(errors)}. No output written.')
    dpo.save(args.output, selected)
    print(f'Saved {len(selected):,} matched no-instruction conversations: {args.output}')
    print(f'Excluded from BOTH no-instruction runs: {dict(errors)}; no replacement conversations.')
    print('Using original base-only Gemini and student answers; no additional score or length-ratio filtering.')


def repair_negatives(args):
    """Retry only length failures; preserve the original cache and sample membership."""
    rows = load(args.sample)
    generated = {r['record_id']: r for r in load(args.generated)}
    if set(generated) != {r['record_id'] for r in rows}:
        raise ValueError('Generated IDs must match the shared sample exactly')
    pending = []
    for row in rows:
        negative = generated[row['record_id']]
        if (negative['source'] != row['source']
                or negative.get('gemini_query') != row.get('gemini_query')
                or negative.get('generation_model') != args.negative_model):
            raise ValueError('Generated prompt, positive, or model does not match the experiment')
        if (negative.get('finish_reason') == 'length'
                and negative.get('generation_error') == 'Incomplete generation: length'):
            # Start from the original sample, without the old error/fingerprint.
            pending.append(row)
        elif (negative.get('generation_error') or negative.get('finish_reason') != 'stop'
                or not dpo.valid_query(negative.get('generated_query'))):
            raise ValueError(f"Non-length generation failure for {row['record_id']}; cannot repair automatically")
    if pending:
        retry_input = args.output.with_name(args.output.stem + '_retry_input.json')
        retry_output = args.output.with_name(args.output.stem + '_retry_generated.json')
        dpo.different_files(retry_input, retry_output, args.output, args.sample, args.generated)
        dpo.save(retry_input, pending)
        print(f'Retrying {len(pending):,}/{len(rows):,} truncated negatives with repetition penalty '
              f'{args.repetition_penalty}; completed responses and original cache are preserved.', flush=True)
        subprocess.run([
            sys.executable, str(Path(dpo.__file__).resolve()), 'generate',
            '--input', str(retry_input), '--output', str(retry_output),
            '--model', str(args.model), '--sft-adapter', str(args.sft_adapter),
            '--model-name', args.negative_model, '--num-gpus', str(args.num_gpus),
            '--max-model-len', str(args.max_length), '--max-new-tokens', str(args.max_new_tokens),
            '--repetition-penalty', str(args.repetition_penalty),
        ], check=True)
        retried = {r['record_id']: r for r in load(retry_output)}
        if set(retried) != {r['record_id'] for r in pending}:
            raise ValueError('Retry output IDs do not match the failed subset')
        failed = []
        for row in pending:
            result = retried[row['record_id']]
            if (result.get('source') != row['source']
                    or result.get('gemini_query') != row['gemini_query']
                    or result.get('generation_model') != args.negative_model):
                raise ValueError('Retry changed a prompt, positive, or model')
            if (result.get('generation_error') or result.get('finish_reason') != 'stop'
                    or not dpo.valid_query(result.get('generated_query'))):
                failed.append(row['record_id'])
            else:
                generated[row['record_id']] = {
                    **result, 'negative_recovery': {
                        'original_finish_reason': 'length',
                        'repetition_penalty': args.repetition_penalty,
                        'original_generated_file': str(args.generated.resolve()),
                    },
                }
        if failed:
            raise ValueError(f'{len(failed):,} negatives still invalid after retry (first: {failed[0]}). '
                             'Training stopped; retry results saved. No pairs were dropped.')
    dpo.save(args.output, (generated[row['record_id']] for row in rows))
    print(f'Saved {len(rows):,} complete negatives: {args.output}; repaired: {len(pending):,}')


def pairs(args):
    rows = load(args.sample)
    generated = None
    if args.generated:
        generated = {r['record_id']: r for r in load(args.generated)}
        if set(generated) != {r['record_id'] for r in rows}:
            raise ValueError('Generated IDs must match the shared sample exactly')
    output = []
    for r in rows:
        negative = r if generated is None else generated[r['record_id']]
        if negative.get('generation_error') or not dpo.valid_query(negative.get('generated_query')):
            raise ValueError(f"Invalid negative for {r['record_id']}; fix generation, do not silently drop pairs")
        if negative['generation_model'] != args.negative_model:
            raise ValueError('Unexpected negative generator')
        if negative['source'] != r['source'] or negative.get('gemini_query') != r['gemini_query']:
            raise ValueError('Generation changed a shared prompt or positive answer')
        if generated is not None and negative.get('finish_reason') != 'stop':
            raise ValueError('Incomplete old-model generation')
        item = dict(record_id=r['record_id'], source_record_id=r.get('source_record_id'),
                    prompt=prompt(r), chosen=[{'role': 'assistant', 'content': r['gemini_query']}],
                    rejected=[{'role': 'assistant', 'content': negative['generated_query']}],
                    user_instruction_mode=r['user_instruction_mode'],
                    chosen_model=r['gemini_model'], rejected_model=negative['generation_model'],
                    experiment='negative_generator_comparison', shared_sample=str(args.sample.resolve()))
        actual_mode = dpo.WITH_INSTRUCTION if dpo.MARKER in r['source']['system_prompt'] else dpo.WITHOUT_INSTRUCTION
        if item['user_instruction_mode'] != actual_mode:
            raise ValueError('Sample instruction condition does not match its prompt')
        for field in ('instruction_applicability_score', 'conditioned_record_id'):
            if field in r:
                item[field] = r[field]
        if 'negative_recovery' in negative:
            item['negative_recovery'] = negative['negative_recovery']
        if 'generation_config' in negative:
            item['rejected_generation_config'] = negative['generation_config']
        validate_pair(item)
        output.append(item)
    # Check both complete branches now; training also enforces strict pair preservation.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    for row in output:
        for branch in ('chosen', 'rejected'):
            ids = tokenizer.apply_chat_template(row['prompt'] + row[branch], tokenize=True,
                                               enable_thinking=False, return_dict=False)
            if len(ids) > args.max_length:
                raise ValueError(f"Overlength {branch}: {row['record_id']}; no pairs written")
    dpo.save(args.output, output)
    identical = sum(dpo.normalized(r['chosen'][0]['content']) == dpo.normalized(r['rejected'][0]['content']) for r in output)
    print(f'Saved {len(output):,} pairs: {args.output}; identical responses retained: {identical}')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('sample')
    s.add_argument('--input', type=Path, required=True)
    s.add_argument('--test', type=Path, required=True)
    s.add_argument('--size', type=dpo.positive, default=10000)
    s.add_argument('--applicability', type=int, choices=range(6), default=2)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--max-new-tokens', type=dpo.positive, default=512)
    s.set_defaults(function=sample)
    n = sub.add_parser('sample-noinst')
    n.add_argument('--input', type=Path, required=True)
    n.add_argument('--conditioned-sample', type=Path, required=True)
    n.add_argument('--max-new-tokens', type=dpo.positive, default=512)
    n.set_defaults(function=sample_noinst)
    p = sub.add_parser('pairs')
    p.add_argument('--sample', type=Path, required=True)
    p.add_argument('--generated', type=Path)
    p.set_defaults(function=pairs)
    repair = sub.add_parser('repair-negatives', help='Retry truncated negatives without dropping shared samples')
    repair.add_argument('--sample', type=Path, required=True)
    repair.add_argument('--generated', type=Path, required=True)
    repair.add_argument('--sft-adapter', type=Path, required=True)
    repair.add_argument('--num-gpus', type=dpo.positive, default=4)
    repair.add_argument('--max-new-tokens', type=dpo.positive, default=512)
    repair.add_argument('--repetition-penalty', type=float, default=1.1)
    repair.set_defaults(function=repair_negatives)
    for cmd in (s, n, p, repair):
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--model', type=Path, required=True)
        cmd.add_argument('--negative-model', required=True)
        cmd.add_argument('--max-length', type=dpo.positive, default=4096)
    args = parser.parse_args()
    try:
        inputs = [getattr(args, name, None) for name in ('input', 'sample', 'generated', 'test', 'conditioned_sample')]
        dpo.different_files(args.output, *(p for p in inputs if p is not None))
        if args.command in ('sample', 'sample-noinst'):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                args.function(args)
        else:
            args.function(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
