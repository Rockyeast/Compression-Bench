"""Task-owned quality-penalized PPL times decode-latency scoring; review required."""
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from inference_runtime import SubmittedRuntime, review_metadata
from ppl_benchmark import validate_result as validate_ppl_result, SCORING_PROTOCOL

ASSETS_DIR = Path(__file__).resolve().parent / 'evaluator-assets'


def config():
    return json.loads((ASSETS_DIR / 'decode_config.json').read_text())


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


def benchmark(submission, tokenizer_dir, *, timeout=600, samples_output=None):
    cfg = config()
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid',
                                      '--format=csv,noheader'], text=True).strip().splitlines()
    if len(devices) != 1:
        raise ValueError('decode measurement requires exactly one visible GPU')
    if version('vllm') != cfg['vllm_version']:
        raise ValueError('decode measurement requires the pinned vLLM environment')
    data = (ASSETS_DIR / 'decode_inputs.json').read_bytes()
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
              'concurrency': 1, 'warmup_requests': 1, 'measured_requests': len(rows),
              'timing_scope': 'parent first-single-token to final-token receipt; '
                              'excludes TTFT, includes stream delivery; speculative decoding prohibited',
              'engine_startup_seconds': startup, 'warmup_seconds': warmup_seconds,
              'warmup_decode_seconds': warmup['decode_seconds'], 'requests': rows,
              **summarize(rows, cfg)}
    return result


def product_metrics(ppl, seconds):
    """Compute raw PPL * seconds before final_score applies the quality policy.

    reward = 1/(1 + C) is solely a bounded monotone encoding for Harbor's
    existing 0..1 result contract. It is not a Reference-relative score.
    """
    for name, value in [('ppl', ppl), ('decode_latency_seconds', seconds)]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if ppl < 1:
        raise ValueError('PPL must be at least 1')
    product = ppl * seconds
    if not math.isfinite(product):
        raise ValueError('non-finite PPL-latency product')
    return {'ppl': ppl, 'decode_latency_seconds': seconds,
            'ppl_latency_product': product, 'reward': 1.0 / (1.0 + product),
            'reward_mapping': '1 / (1 + ppl * decode_latency_seconds)',
            'quality_gate_enabled': False, 'quality_pass': None}


def final_score(ppl_result, decode_result, *, public=False, ppl_assets=None):
    cfg = config()
    if ppl_result.get('valid') is not True:
        raise ValueError('no valid quality measurement')
    validate_ppl_result(ppl_result, ppl_assets or ASSETS_DIR, expected_split='public' if public else 'hidden')
    if decode_result.get('valid') is not True or decode_result.get('complete') is not True:
        raise ValueError('no valid complete decode measurement')
    if decode_result.get('measurement_protocol') != cfg['measurement_protocol']:
        raise ValueError('decode result uses the wrong protocol')
    summary = summarize(decode_result['requests'], cfg)
    gate = quality_gate(ppl_result['ppl'], ppl_assets or ASSETS_DIR)
    metrics = product_metrics(ppl_result['ppl'], summary['median_round_mean_seconds'])
    metrics.update(gate)
    adjusted = metrics['ppl_latency_product'] * gate['quality_penalty_multiplier']
    if not math.isfinite(adjusted):
        raise ValueError('non-finite penalized PPL-latency product')
    metrics['penalized_ppl_latency_product'] = adjusted
    metrics['reward'] = 1.0 / (1.0 + adjusted) if gate['quality_pass'] else 0.0
    metrics['reward_mapping'] = ('0 if ppl > 1.20 * split_bf16_ppl else '
                                 '1 / (1 + ppl * decode_latency_seconds * quality_penalty_multiplier)')
    return {'valid': True, **review_metadata(), 'scoring_version': SCORING_PROTOCOL,
            **{key:ppl_result[key] for key in ('ppl_protocol','ppl_split','predicted_tokens')},
            'public_only': public, 'is_official_score': not public, **summary,
            **metrics}


def quality_gate(ppl, assets):
    """Same-split BF16: linear 1..1.5 cost penalty over 10..20%, inclusive cap."""
    cfg = json.loads((Path(assets) / 'ppl_config.json').read_text())
    baseline = cfg.get('bf16_ppl')
    if (cfg.get('baseline_status') != 'measured_new_protocol'
            or type(baseline) not in (int, float) or not math.isfinite(baseline) or baseline < 1):
        raise ValueError('quality gate requires a measured task-owned BF16 PPL baseline')
    if type(ppl) not in (int, float) or not math.isfinite(ppl) or ppl < 1:
        raise ValueError('quality gate requires finite PPL >= 1')
    limit = baseline * 1.20
    start = baseline * 1.10
    if not math.isfinite(limit):
        raise ValueError('non-finite BF16-relative quality limit')
    passed = ppl <= limit
    # Direct threshold comparisons preserve the inclusive floating-point boundaries.
    # Rejected candidates report the capped multiplier too, but always earn zero.
    fraction = min(1.0, max(0.0, (ppl - start) / (limit - start)))
    multiplier = 1.0 + 0.5 * fraction
    return dict(quality_gate_enabled=True, quality_pass=passed, ranking_eligible=passed,
                quality_bf16_ppl=baseline, quality_max_ratio=1.20, quality_ppl_limit=limit,
                quality_gray_start_ratio=1.10, quality_gray_start_ppl=start,
                quality_max_penalty_multiplier=1.5, quality_penalty_multiplier=multiplier,
                ppl_increase_percent=100*(ppl/baseline-1),
                quality_rejection_reason=None if passed else 'ppl_exceeds_bf16_by_more_than_20_percent')
