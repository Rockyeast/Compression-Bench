"""Task-owned fixed C4 blocks; shared by public Checker and hidden Verifier."""
import json
import math
from pathlib import Path

PROTOCOL = 'c4-gptq-256x2048-v1'
SCORING_PROTOCOL = 'ppl-latency-product-c4-bf16-gray15-cap20-v4'
BLOCKS = 256
CONTEXT = 2048
SCORED_TOKENS = BLOCKS * (CONTEXT - 1)


def load_inputs(assets, tokenizer_dir, *, expected_split=None):
    assets, tokenizer_dir = Path(assets), Path(tokenizer_dir)
    cfg = json.loads((assets / 'ppl_config.json').read_text())
    if (cfg.get('protocol') != PROTOCOL or cfg.get('blocks') != BLOCKS
            or cfg.get('context_length') != CONTEXT or cfg.get('scored_tokens') != SCORED_TOKENS
            or cfg.get('input_tokens') != BLOCKS * CONTEXT):
        raise ValueError('wrong fixed C4 PPL protocol')
    if expected_split is not None and cfg.get('split') != expected_split:
        raise ValueError('wrong public/hidden PPL split')
    data = (assets / 'ppl_inputs.json').read_bytes()
    rows = json.loads(data)
    if (len(rows) != BLOCKS or any(len(row) != CONTEXT for row in rows)
            or any(type(t) is not int or not 0 <= t < cfg['vocab_size'] for row in rows for t in row)):
        raise ValueError('invalid C4 input blocks or token IDs')
    return cfg, rows


def measure(submission, assets, tokenizer_dir, runtime_class, *, timeout=600, expected_split=None):
    cfg, rows = load_inputs(assets, tokenizer_dir, expected_split=expected_split)
    windows = []
    # Match the validated experiment: batches of 16, engine capacity 32, no overlap.
    with runtime_class(submission, max_model_len=2049, max_num_seqs=32,
                       gpu_memory_utilization=0.45, timeout=timeout) as runtime:
        for offset in range(0, BLOCKS, 16):
            batch = rows[offset:offset + 16]
            outputs = runtime.token_logprobs(batch)
            if len(outputs) != len(batch):
                raise ValueError('missing PPL probability rows')
            for i, (ids, logprobs) in enumerate(zip(batch, outputs, strict=True)):
                if len(logprobs) != len(ids):
                    raise ValueError('misaligned PPL probabilities')
                values = logprobs[1:]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in values):
                    raise ValueError('invalid PPL log probabilities')
                windows.append({'block': offset+i, 'nll': -sum(values), 'scored_tokens': CONTEXT-1})
    nll = sum(w['nll'] for w in windows)
    ppl = math.exp(nll / SCORED_TOKENS)
    if not math.isfinite(ppl) or ppl < 1:
        raise ValueError('PPL must be finite and >= 1')
    return dict(valid=True, ppl=ppl, nll=nll, predicted_tokens=SCORED_TOKENS,
                input_tokens=BLOCKS*CONTEXT, blocks=BLOCKS, context_length=CONTEXT,
                stride=CONTEXT, batch_size=16, max_num_seqs=32, gpu_memory_utilization=0.45,
                ppl_protocol=PROTOCOL,
                ppl_split=cfg['split'],
                windows=windows, quality_gate_enabled=False)


def validate_result(result, assets, *, expected_split):
    """Do not score stale WikiText outputs or mix public and hidden measurements."""
    cfg = json.loads((Path(assets) / 'ppl_config.json').read_text())
    if (cfg.get('protocol') != PROTOCOL or cfg.get('split') != expected_split
            or result.get('ppl_protocol') != PROTOCOL
            or result.get('ppl_split') != expected_split
            or result.get('predicted_tokens') != SCORED_TOKENS):
        raise ValueError('PPL result uses the wrong protocol, split or token count')
