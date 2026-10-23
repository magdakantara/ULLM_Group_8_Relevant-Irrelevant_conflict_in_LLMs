"""Run local Hugging Face inference from the shared JSON; no paid API calls."""
import argparse
import json
from importlib.metadata import version
from output_format import FORMAT_VERSION, RESPONSE_SCHEMA, new_prefix_constraint, prepare_tokenizer
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from common import (ROOT, DEFAULT_DATASET, PROMPTS, read, write, rows, fingerprint,
                    load_dataset, select_examples, messages, parse_response, reference_snapshot)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(DEFAULT_DATASET))
    parser.add_argument('--config', default=str(ROOT / 'config/experiment_long.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--prompt', choices=list(PROMPTS), default='baseline')
    parser.add_argument('--example-id', help='Run only this exact example ID.')
    parser.add_argument('--owner', help='Owner initial: M, B, E, O or D.')
    parser.add_argument('--contract', help='Distinctive contract name or example ID prefix, e.g. Driven.')
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument('--limit', type=int, help='Maximum number of examples to run.')
    scope.add_argument('--all', action='store_true', help='Run every example matching the filters.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Inspect inputs without model libraries or downloads.')
    mode.add_argument('--preflight', action='store_true', help='Count tokens using only the tokenizer and model config.')
    return parser.parse_args()


def append_result(path, result):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    args = arguments()
    data = load_dataset(args.data)
    pool = data
    if args.example_id:
        pool = [r for r in data if r['example_id'] == args.example_id]
    chosen = select_examples(pool, args.owner, args.contract, args.limit)
    config = read(args.config)
    for key in ['max_input_tokens', 'max_new_tokens', 'retry_max_new_tokens']:
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if config['retry_max_new_tokens'] < config['max_new_tokens']:
        raise ValueError('Retry output limit must be at least the initial output limit.')
    prompt = (ROOT / 'prompts' / PROMPTS[args.prompt]).read_text(encoding='utf-8')
    signature = fingerprint({'dataset': data, 'selected_ids': [r['example_id'] for r in chosen],
                             'config': config, 'prompt': prompt,
                             'output_format': FORMAT_VERSION, 'response_schema': RESPONSE_SCHEMA})
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / 'run_info.json'
    dest = folder / 'responses.jsonl'
    previous = read(meta) if meta.exists() else None
    if previous and previous['signature'] != signature:
        raise ValueError('Data, prompt or selected examples changed. Choose a new --out directory.')
    if dest.exists() and previous is None:
        raise ValueError('Responses exist without run metadata. Use a new output directory.')
    if args.dry_run:
        write(folder / 'dry_run.json', {'examples': len(chosen), 'prompt': args.prompt,
              'example_ids': [r['example_id'] for r in chosen],
              'first_messages': messages(prompt, chosen[0]),
              'note': 'No model loaded. Only Source A, Source B and question are sent as user input.'})
        print(f'Dry run: {len(chosen)}', flush=True)
        return

    completed = rows(dest) if dest.exists() else []
    completed_ids = [r['example_id'] for r in completed]
    if len(set(completed_ids)) != len(completed_ids) or not set(completed_ids) <= {r['example_id'] for r in chosen}:
        raise ValueError('Duplicate or unknown IDs in responses.jsonl.')
    pending = [r for r in chosen if r['example_id'] not in set(completed_ids)]
    if not pending and not args.preflight:
        print('All selected examples are already recorded. No model loaded.')
        return

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    # Freeze the resolved model revision across baseline/explained runs in this project.
    lock_path = ROOT / 'config/model_revision_lock.json'
    lock = read(lock_path) if lock_path.exists() else None
    requested_revision = config.get('revision', 'main')
    if lock and (lock['model'] != config['model'] or lock['requested_revision'] != requested_revision):
        raise ValueError('Model configuration differs from model_revision_lock.json. Use a separate project copy for another model.')
    revision = lock['resolved_revision'] if lock else requested_revision
    model_config = AutoConfig.from_pretrained(config['model'], revision=revision)
    from long_context import configure_long_context
    configure_long_context(model_config, config)
    resolved = getattr(model_config, '_commit_hash', None)
    if not resolved:
        raise ValueError('Could not resolve an immutable Hugging Face model revision.')
    if previous and previous['resolved_revision'] != resolved:
        raise ValueError('Model revision changed. Choose a new output directory.')
    if not lock:
        write(lock_path, {'model': config['model'], 'requested_revision': requested_revision, 'resolved_revision': resolved})
    tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=resolved)
    context_limit = min(config['max_input_tokens'], model_config.max_position_embeddings)
    if args.preflight:
        records = []
        for row in chosen:
            tokens = tokenizer.apply_chat_template(messages(prompt, row), tokenize=True, add_generation_prompt=True)
            records.append({'example_id': row['example_id'], 'input_tokens': len(tokens),
                            'fits': len(tokens) + config['retry_max_new_tokens'] <= context_limit})
        write(folder / 'preflight.json', {'examples': len(records), 'context_limit': context_limit,
              'reserved_output_tokens': config['retry_max_new_tokens'], 'resolved_revision': resolved,
              'prompt': args.prompt, 'signature': signature, 'records': records,
              'dataset_sha256': fingerprint(data), 'config': config, 'system_prompt': prompt})
        print(f"Preflight: {sum(r['fits'] for r in records)}/{len(records)} fit; no model weights loaded.")
        return

    format_library_version = version('lm-format-enforcer')
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from transformers import AutoModelForCausalLM, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Use a Snellius GPU session; staging supports --dry-run and --preflight.')
    snapshot = reference_snapshot(chosen)
    info = {'signature': signature, 'dataset_sha256': fingerprint(data), 'dataset_path': str(Path(args.data).resolve()),
            'selected_ids': [r['example_id'] for r in chosen], 'config': config,
            'prompt_name': args.prompt, 'system_prompt': prompt, 'resolved_revision': resolved,
            'reference_sha256': fingerprint(snapshot), 'python': platform.python_version(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(0), 'context_limit': context_limit,
            'note': 'Evaluation run, not training. No source-review approval field is used. No development/test split is inferred.'}
    info['output_format'] = FORMAT_VERSION
    info['response_schema'] = RESPONSE_SCHEMA
    info['lm_format_enforcer'] = format_library_version
    if previous:
        for key in ['python', 'torch', 'transformers', 'lm_format_enforcer', 'output_format']:
            if previous.get(key) != info[key]:
                raise ValueError(f'{key} changed; use a new output directory.')
        if read(folder / 'reference_snapshot.json') != snapshot:
            raise ValueError('Saved reference snapshot differs from the current dataset.')
    else:
        write(folder / 'reference_snapshot.json', snapshot)
        write(meta, info)
        (folder / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    # Keep all weights on the allocated GPU: no silent CPU offload.
    model = AutoModelForCausalLM.from_pretrained(config['model'], revision=resolved,
             torch_dtype=getattr(torch, config['dtype']), device_map={'': 0},
             config=model_config, attn_implementation='sdpa')
    model.eval()
    format_tokenizer = prepare_tokenizer(tokenizer)
    for row in pending:
        set_seed(config['seed'])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        print(f"Starting {row['example_id']} (full source texts)", flush=True)
        start = time.monotonic()
        result = {'example_id': row['example_id'], 'status': 'ok', 'response': '', 'parsed': None}
        encoded = output = ids = None
        try:
            encoded = tokenizer.apply_chat_template(messages(prompt, row), tokenize=True,
                      add_generation_prompt=True, return_tensors='pt', return_dict=True)
            n = encoded['input_ids'].shape[1]
            result['input_tokens'] = n
            if n + config['retry_max_new_tokens'] > context_limit:
                result.update(status='context_overflow', context_limit=context_limit)
            else:
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                attempts = []
                for cap in dict.fromkeys([config['max_new_tokens'], config['retry_max_new_tokens']]):
                    prefix_constraint = new_prefix_constraint(format_tokenizer)
                    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        output = model.generate(**encoded, do_sample=False, max_new_tokens=cap,
                                                pad_token_id=tokenizer.eos_token_id,
                                                prefix_allowed_tokens_fn=prefix_constraint, logits_to_keep=1)
                    ids = output[0, n:]
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
                    eos = eos if isinstance(eos, list) else [eos]
                    ended = len(ids) > 0 and int(ids[-1]) in eos
                    attempts.append({'cap': cap, 'output_tokens': len(ids), 'text': text, 'ended': ended})
                    output = ids = None
                    if ended:
                        break
                parsed = parse_response(text)
                result.update(response=text, parsed=parsed, attempts=attempts,
                              status='truncated' if not ended else 'ok' if parsed else 'format_error')
                if parsed:
                    result['quote_checks'] = {key: (parsed[key] in row[source] if parsed[key] else None)
                        for key, source in [('evidence_A', 'source_A'), ('evidence_B', 'source_B')]}
        except torch.cuda.OutOfMemoryError:
            result.update(status='out_of_memory', response='', parsed=None)
        finally:
            encoded = output = ids = None
            if result['status'] == 'out_of_memory':
                torch.cuda.empty_cache()
        torch.cuda.synchronize()
        result['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
        result['peak_gpu_reserved_gib'] = torch.cuda.max_memory_reserved() / 2**30
        result['seconds'] = time.monotonic() - start
        append_result(dest, result)
        print(row['example_id'], result['status'],
              f"{result['seconds']:.1f}s; peak allocated {result['peak_gpu_allocated_gib']:.2f} GiB", flush=True)
    print(f'Results saved to {dest}')


if __name__ == '__main__':
    main()
