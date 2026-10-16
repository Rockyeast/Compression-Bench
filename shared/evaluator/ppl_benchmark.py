"""Task-owned fixed C4 blocks; shared by public Checker and hidden Verifier."""
import json
import math
from pathlib import Path

# Protocols fix the prefix length; task-owned assets select one per call.
PREFIX_TOKENS = {
    'c4-gptq-256x2048-v1': 0,
    'c4-gemma-chat-256x2048-v1': 19,
    'c4-gemma31-chat-256x2048-v1': 19,
    'c4-qwen35-chat-256x2048-v1': 18,
}
BLOCKS = 256
CONTEXT = 2048


def load_config(assets, *, expected_split=None):
    cfg = json.loads((Path(assets) / 'ppl_config.json').read_text())
    prefix = PREFIX_TOKENS.get(cfg.get('protocol'))
    if prefix is None:
        raise ValueError('unknown fixed C4 PPL protocol')
    score_start = max(1, prefix)
    if (cfg.get('blocks') != BLOCKS or cfg.get('context_length') != CONTEXT
            or cfg.get('scored_tokens') != BLOCKS * (CONTEXT - score_start)
            or cfg.get('input_tokens') != BLOCKS * CONTEXT):
        raise ValueError('wrong fixed C4 PPL protocol')
    if expected_split is not None and cfg.get('split') != expected_split:
        raise ValueError('wrong public/hidden PPL split')
    return cfg, score_start


def load_inputs(assets, *, expected_split=None):
    cfg, score_start = load_config(assets, expected_split=expected_split)
    rows = json.loads((Path(assets) / 'ppl_inputs.json').read_bytes())
    if (len(rows) != BLOCKS or any(len(row) != CONTEXT for row in rows)
            or any(type(t) is not int or not 0 <= t < cfg['vocab_size'] for row in rows for t in row)):
        raise ValueError('invalid C4 input blocks or token IDs')
    prefix = PREFIX_TOKENS[cfg['protocol']]
    if prefix and any(row[:prefix] != cfg.get('prefix_tokens') for row in rows):
        raise ValueError('wrong model chat prefix')
    return cfg, rows, score_start


def measure(submission, assets, tokenizer_dir, runtime_class, *, timeout=600, expected_split=None):
    cfg, rows, score_start = load_inputs(assets, expected_split=expected_split)
    scored_tokens = cfg['scored_tokens']
    windows = []
    # Match the validated experiment: batches of 16, engine capacity 32, no overlap.
    with runtime_class(submission, max_model_len=2049, max_num_seqs=32,
                       gpu_memory_utilization=0.9, timeout=timeout) as runtime:
        for offset in range(0, BLOCKS, 16):
            batch = rows[offset:offset + 16]
            outputs = runtime.token_logprobs(batch)
            if len(outputs) != len(batch):
                raise ValueError('missing PPL probability rows')
            for i, (ids, logprobs) in enumerate(zip(batch, outputs, strict=True)):
                if len(logprobs) != len(ids):
                    raise ValueError('misaligned PPL probabilities')
                values = logprobs[score_start:]
                if any(type(v) not in (int, float) or not math.isfinite(v) or v > 1e-5 for v in values):
                    raise ValueError('invalid PPL log probabilities')
                windows.append({'block': offset+i, 'nll': -sum(values), 'scored_tokens': CONTEXT-score_start})
    nll = sum(w['nll'] for w in windows)
    ppl = math.exp(nll / scored_tokens)
    if not math.isfinite(ppl) or ppl < 1:
        raise ValueError('PPL must be finite and >= 1')
    return dict(valid=True, model_memory=runtime.memory_report(), ppl=ppl, nll=nll, predicted_tokens=scored_tokens,
                input_tokens=BLOCKS*CONTEXT, blocks=BLOCKS, context_length=CONTEXT,
                stride=CONTEXT, batch_size=16, max_num_seqs=32, gpu_memory_utilization=0.9,
                ppl_protocol=cfg['protocol'],
                ppl_split=cfg['split'],
                windows=windows, quality_gate_enabled=False)


def validate_result(result, assets, *, expected_split):
    """Check results against this call's task-owned protocol, split and token count."""
    cfg, _ = load_config(assets, expected_split=expected_split)
    if (result.get('ppl_protocol') != cfg['protocol']
            or result.get('ppl_split') != expected_split
            or result.get('predicted_tokens') != cfg['scored_tokens']):
        raise ValueError('PPL result uses the wrong protocol, split or token count')
