"""Shared streamed decode measurement and timing validation."""
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from inference_runtime import SubmittedRuntime, review_metadata

def config(assets):
    return json.loads((Path(assets) / 'decode_config.json').read_text())


def summarize(rows, cfg):
    expected = {(r, p) for r in range(cfg['rounds']) for p in range(cfg['requests_per_round'])}
    if len(rows) != len(expected) or {(r['round'], r['request']) for r in rows} != expected:
        raise ValueError('missing or duplicate decode requests')
    for row in rows:
        times = row['arrival_seconds']
        sizes, chunk_times = row['chunk_sizes'], row['chunk_arrival_seconds']
        if (not isinstance(sizes, list) or not sizes or sizes[0] != 1
                or any(type(n) is not int or n != 1 for n in sizes)
                or sum(sizes) != cfg['output_tokens']
                or not isinstance(chunk_times, list) or len(sizes) != len(chunk_times)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in chunk_times)
                or times != [t for n, t in zip(sizes, chunk_times) for _ in range(n)]):
            raise ValueError('invalid single-token events or receipt timestamps')
        if (row['input_tokens'] != cfg['input_tokens'] or row['output_tokens'] != cfg['output_tokens']
                or row['decode_intervals'] != cfg['output_tokens'] - 1
                or len(times) != cfg['output_tokens']
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(a > b for a, b in zip(times, times[1:]))):
            raise ValueError('invalid streamed token count or arrival timestamps')
        elapsed = times[-1] - times[0]
        if elapsed <= 0 or not math.isclose(row['decode_seconds'], elapsed, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('decode time does not match first-to-last token interval')
    means = [statistics.mean(row['decode_seconds'] for row in rows if row['round'] == r)
             for r in range(cfg['rounds'])]
    median = statistics.median(means)
    return {'round_mean_seconds': means, 'median_round_mean_seconds': median,
            'decode_tokens_per_second': (cfg['output_tokens'] - 1) / median,
            'round_cv': statistics.pstdev(means) / statistics.mean(means),
            'round_spread_fraction': (max(means) - min(means)) / median}


def benchmark(submission, tokenizer_dir, *, assets, timeout=600, samples_output=None):
    cfg = config(assets)
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid',
                                      '--format=csv,noheader'], text=True).strip().splitlines()
    if len(devices) != 1:
        raise ValueError('decode measurement requires exactly one visible GPU')
    if version('vllm') != cfg['vllm_version']:
        raise ValueError('decode measurement requires the pinned vLLM environment')
    data = (Path(assets) / 'decode_inputs.json').read_bytes()
    prompts = json.loads(data)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    vocab_size = len(tokenizer)
    measured = prompts['measured']
    if len(measured) != cfg['requests_per_round']:
        raise ValueError('wrong number of decode prompts')
    for ids in [prompts['warmup'], *measured]:
        if (len(ids) != cfg['input_tokens']
                or any(type(t) is not int or not 0 <= t < vocab_size for t in ids)):
            raise ValueError('invalid fixed decode prompt')
    rows = []
    log = Path(samples_output).open('x') if samples_output is not None else None
    try:
        started = time.perf_counter()
        with SubmittedRuntime(submission, max_model_len=cfg['max_model_len'],
                              max_num_seqs=cfg['max_num_seqs'],
                              gpu_memory_utilization=cfg['gpu_memory_utilization'], timeout=timeout) as engine:
            startup = time.perf_counter() - started
            warmup_start = time.perf_counter()
            warmup = engine.measure_decode(prompts['warmup'], max_new_tokens=cfg['output_tokens'],
                                            vocab_size=vocab_size)
            warmup_seconds = time.perf_counter() - warmup_start
            for r in range(cfg['rounds']):
                for p, ids in enumerate(measured):
                    sample = engine.measure_decode(ids, max_new_tokens=cfg['output_tokens'],
                                                   vocab_size=vocab_size)
                    row = {'round': r, 'request': p, 'input_tokens': len(ids),
                           'output_tokens': len(sample['token_ids']),
                           **{k: v for k, v in sample.items() if k != 'token_ids'}}
                    rows.append(row)
                    if log:
                        log.write(json.dumps(row) + '\n')
                        log.flush()
                    print(f'decode round {r+1}/3 request {p+1}/3: {row["decode_seconds"]:.6f}s', flush=True)
    finally:
        if log:
            log.close()
    result = {'valid': True, 'complete': True, **review_metadata(),
              'measurement_protocol': cfg['measurement_protocol'],
              'device': devices[0],
              'input_tokens': cfg['input_tokens'], 'output_tokens': cfg['output_tokens'],
              'gpu_memory_utilization': cfg['gpu_memory_utilization'],
              'concurrency': 1, 'warmup_requests': 1, 'measured_requests': len(rows),
              'timing_scope': 'parent first-single-token to final-token receipt; '
                              'excludes TTFT, includes stream delivery; speculative decoding prohibited',
              'engine_startup_seconds': startup, 'warmup_seconds': warmup_seconds,
              'warmup_decode_seconds': warmup['decode_seconds'], 'requests': rows,
              'model_memory': engine.memory_report(),
              **summarize(rows, cfg)}
    return result
