"""Offline block-influence ranking inspired by ShortGPT; does not delete layers.

Own implementation of mean(1 - cosine(block_input, block_output)). Supply the
actual text decoder ModuleList path; no model-name or Qwen-specific walker.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))
from common import load_model, read_blocks


def score_layers(model, layers_path, blocks, device='cpu'):
    import torch
    from torch import nn
    from torch.nn.functional import cosine_similarity
    layers = model.get_submodule(layers_path)
    if not isinstance(layers, nn.ModuleList) or not len(layers):
        raise ValueError('--layers-path must identify a nonempty decoder ModuleList')
    sums, counts, hooks = [0.] * len(layers), [0] * len(layers), []
    def hook_for(index):
        def hook(module, args, kwargs, output):
            x = args[0] if args else kwargs.get('hidden_states')
            y = output[0] if isinstance(output, tuple) else output
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or x.shape != y.shape:
                raise ValueError('Each block must expose matching input/output hidden tensors')
            values = 1 - cosine_similarity(x.detach().float(), y.detach().float(), dim=-1)
            if not torch.isfinite(values).all():
                raise ValueError('Non-finite block influence')
            sums[index] += values.double().sum().item()
            counts[index] += values.numel()
        return hook
    was_training = model.training
    try:
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(hook_for(i), with_kwargs=True))
        model.eval()
        with torch.no_grad():
            for row in blocks:
                ids = torch.tensor([row], device=device)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    if any(n == 0 for n in counts):
        raise ValueError('Some blocks were not executed; inspect the selected path')
    return [{'layer': i, 'module': f'{layers_path}.{i}', 'mean_block_influence': s/n,
             'token_positions': n} for i, (s, n) in enumerate(zip(sums, counts))]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--calibration', required=True, type=Path, help='JSON token blocks from public training data')
    p.add_argument('--layers-path', required=True, help='Text block list, e.g. model.layers; inspect named_modules()')
    p.add_argument('--output', required=True, type=Path, help='Fresh JSON report, outside model/input directories')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32','bfloat16','float16'], default='bfloat16')
    a = p.parse_args()
    if a.output.exists():
        p.error('Output exists; choose a fresh report path')
    model = load_model(a.model, a.device, a.dtype)
    rows = score_layers(model, a.layers_path, read_blocks(a.calibration), a.device)
    report = {'method': 'mean block influence', 'layers': rows,
              'ranking_lowest_first': sorted(range(len(rows)), key=lambda i: rows[i]['mean_block_influence']),
              'scope': 'Calibration diagnostic only; not a pruned model or quality/speed result'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
