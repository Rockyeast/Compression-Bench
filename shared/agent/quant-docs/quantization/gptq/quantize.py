"""Offline accelerated GPTQ entry for local models. Run with /opt/gptq/bin/python."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import time


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New work directory; exported weights go in model/')
    p.add_argument('--calibration', type=Path, required=True, help='Public train data: text parquet or token-block JSON/NPY')
    p.add_argument('--targets', nargs='+', required=True, help='Exact module names or re: patterns for text Linear modules')
    p.add_argument('--sequential-targets', nargs='+', required=True, help='Decoder block class names supported by upstream tracing')
    p.add_argument('--ignore', nargs='*', default=['lm_head'])
    p.add_argument('--bits', type=int, choices=[2, 3, 4, 8], default=4)
    p.add_argument('--group-size', type=int, choices=[32, 64, 128], default=128)
    p.add_argument('--dampening-frac', type=float, default=0.01)
    p.add_argument('--asymmetric', action='store_true')
    p.add_argument('--batch', type=int, default=1, help='Calibration text batch size (default: 1)')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    return p


def validate_args(p, a):
    if a.batch < 1 or a.samples < 1 or a.seqlen < 2:
        p.error('batch/samples must be positive and seqlen >= 2')
    if not math.isfinite(a.dampening_frac) or a.dampening_frac < 0:
        p.error('dampening-frac must be finite and nonnegative')
    if not (a.model / 'config.json').is_file() or not a.calibration.is_file():
        p.error('Supply an existing local model and calibration file')
    if a.output.exists():
        p.error('Output already exists; choose a fresh directory')
    for pattern in a.targets + a.ignore:
        if pattern.startswith('re:'):
            re.compile(pattern[3:])
    return json.loads((a.model / 'config.json').read_text())


def calibration_blocks(path, tokenizer, count, length, seed):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        texts = pq.read_table(path, columns=['text'])['text'].to_pylist()
        ids = tokenizer('\n\n'.join(t for t in texts if t), add_special_tokens=False).input_ids
        rows = [ids[i:i + length] for i in range(0, len(ids) - length + 1, length)]
    elif path.suffix == '.npy':
        import numpy as np
        rows = np.load(path, allow_pickle=False).tolist()
    elif path.suffix == '.json':
        rows = json.loads(path.read_text())
    else:
        raise ValueError('Expected calibration .parquet, .json or .npy')
    if not isinstance(rows, list) or any(
        not isinstance(row, list) or len(row) < length or
        any(type(x) is not int or not 0 <= x < len(tokenizer) for x in row)
        for row in rows
    ):
        raise ValueError('Calibration needs valid token-ID rows at least seqlen long')
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} calibration blocks; requested {count}')
    return [rows[i][:length] for i in random.Random(seed).sample(range(len(rows)), count)]


def matches(name, pattern):
    return bool(re.match(pattern[3:], name)) if pattern.startswith('re:') else name == pattern


def compact_targets(names):
    """Exact finite name set as a factored regex, avoiding O(modules * targets).

    Paths with identical remaining subtrees share a branch. Every component is
    escaped; this does not generalize digit ranges or select unlisted modules.
    """
    if len(names) < 256:
        return list(names)
    trie = {}
    for name in names:
        node = trie
        for part in name.split('.'):
            node = node.setdefault(part, {})
        node[None] = {}

    def render(node, root=False):
        groups = {}
        for part, child in node.items():
            if part is not None:
                suffix = render(child)
                groups.setdefault(suffix, []).append(re.escape(part))
        branches = [''] if None in node else []
        for suffix, parts in groups.items():
            prefix = parts[0] if len(parts) == 1 else '(?:' + '|'.join(parts) + ')'
            branches.append(('' if root else r'\.') + prefix + suffix)
        return branches[0] if len(branches) == 1 else '(?:' + '|'.join(branches) + ')'

    return ['re:^' + render(trie, root=True) + '$']


def add_shared_kv_targets(config, resolved_targets=None):
    """Add runtime QKV aliases for Gemma4's quantized, tied K/V projections.

    vLLM 0.28 matches all three unfused names when selecting a QKV scheme.
    Full-attention K=V layers have no independent v_proj in the checkpoint.
    An explicit fused target selects the existing Q/K scheme; the model loader
    duplicates K into V. This changes metadata only, not weights or precision.
    """
    text = config.get('text_config') or config
    if text.get('model_type') != 'gemma4_text' or not text.get('attention_k_eq_v'):
        return []
    quant = config.get('quantization_config') or {}
    if quant.get('quant_method') != 'compressed-tensors':
        return []
    groups = quant.get('config_groups', {})
    added = []
    for index, kind in enumerate(text.get('layer_types', [])):
        if kind != 'full_attention':
            continue
        suffix = f'layers.{index}.self_attn.q_proj'
        for group in groups.values():
            targets = group.get('targets', [])
            # Use the validated exact name set when the recipe is factored.
            # Without that set, do not infer names from arbitrary regex targets.
            queries = [t for t in (resolved_targets if resolved_targets is not None else targets)
                       if not t.startswith('re:') and t.endswith(suffix)
                       and any(matches(t, pattern) for pattern in targets)]
            for query in queries:
                key = query.removesuffix('q_proj') + 'k_proj'
                if not any(matches(key, pattern) for pattern in targets):
                    raise ValueError(f'Shared K/V fused attention requires Q and K '
                                     f'in the same quantization group: {query}')
                fused = query.removesuffix('q_proj') + 'qkv_proj'
                if fused not in targets:
                    targets.append(fused)
                    added.append(fused)
    return added


def restore_export_targets(quant_config, resolved_targets):
    """Keep recipe regex optimization out of runtime model-name mapping."""
    for group in quant_config['config_groups'].values():
        patterns = group['targets']
        group['targets'] = [name for name in resolved_targets
                            if any(matches(name, pattern) for pattern in patterns)]


def main():
    p = make_parser()
    a = p.parse_args()
    original = validate_args(p, a)
    if original.get('quantization_config') or (original.get('text_config') or {}).get('quantization_config'):
        p.error('Start from floating-point model weights')
    a.model, a.output, a.calibration = (x.resolve() for x in (a.model, a.output, a.calibration))
    for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    os.environ['LLMCOMPRESSOR_DISABLE_GPTQ_TRITON'] = '0'
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.gptq import gptq_quantize
    if 'batched_quantization' not in GPTQModifier.model_fields or not hasattr(gptq_quantize, '_gptq_block_update_triton'):
        raise RuntimeError('Use the pinned /opt/gptq environment with batched and Triton GPTQ support')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(a.seed)
    torch.set_num_threads(8)
    start = time.monotonic()
    config = AutoConfig.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    cls = AutoModelForImageTextToText if getattr(config, 'vision_config', None) is not None else AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    blocks = calibration_blocks(a.calibration, tokenizer, a.samples, a.seqlen, a.seed)
    dataset = [{'input_ids': torch.tensor(row), 'attention_mask': torch.ones(len(row), dtype=torch.long)} for row in blocks]
    loader = DataLoader(dataset, batch_size=a.batch, shuffle=False)
    # Linearize fused MoE parameters before resolving target patterns. Otherwise
    # nn.Linear traversal can silently omit every routed expert.
    from llmcompressor.utils import load_context
    with load_context(cls):
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True,
                                   trust_remote_code=False, low_cpu_mem_usage=True).eval()
    # Resolve explicit patterns to exact names before handing them to upstream.
    targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
               and any(matches(name, t) for t in a.targets)
               and not any(matches(name, t) for t in a.ignore)]
    if not targets:
        raise ValueError('No Linear modules matched targets after exclusions')
    if any(re.search(r'(^|\.)(visual|vision_model|vision_tower|vision_encoder|vision)(\.|$)', name) for name in targets):
        raise ValueError('Select text-only targets; vision modules must remain unchanged')
    weights = dict(num_bits=a.bits, type='int', symmetric=not a.asymmetric,
                   strategy='group', group_size=a.group_size, actorder='static', dynamic=False)
    recipe_targets = compact_targets(targets)
    recipe = GPTQModifier(config_groups={'weights': dict(targets=recipe_targets, weights=weights)},
                          dampening_frac=a.dampening_frac, block_size=128, batched_quantization='auto')
    a.output.mkdir(parents=True)
    record = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    record.update(resolved_targets=targets, recipe_targets=recipe_targets, batched_quantization='auto', triton='enabled when supported',
                  versions={k: importlib.metadata.version(k) for k in ('torch', 'transformers', 'llmcompressor', 'compressed-tensors', 'triton')})
    (a.output / 'request.json').write_text(json.dumps(record, indent=2) + '\n')
    os.chdir(a.output)
    qstart = time.monotonic()
    oneshot(model=model, tokenizer=tokenizer, dataset=loader, recipe=recipe,
            num_calibration_samples=a.samples, max_seq_length=a.seqlen, batch_size=a.batch,
            shuffle_calibration_samples=False, pipeline='sequential',
            sequential_targets=a.sequential_targets, sequential_offload_device='cpu',
            output_dir=str(a.output / 'model'), save_compressed=True)
    exported = a.output / 'model/config.json'
    saved = json.loads(exported.read_text())
    if not saved.get('quantization_config'):
        raise RuntimeError('Export is missing quantization metadata')
    # HF serialization can rename nested vision model types. Keep source config
    # fields, adding only the quantization metadata produced by the exporter.
    original['quantization_config'] = saved['quantization_config']
    restore_export_targets(original['quantization_config'], targets)
    record['runtime_target_aliases'] = add_shared_kv_targets(original, targets)
    exported.write_text(json.dumps(original, indent=2) + '\n')
    record.update(status='exported; target-runtime reload and Checker required',
                  quantize_export_seconds=time.monotonic()-qstart, total_seconds=time.monotonic()-start)
    (a.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
