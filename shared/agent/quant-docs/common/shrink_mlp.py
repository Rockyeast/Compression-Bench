"""Uniform physical gated-MLP channel reduction; a magnitude baseline, not Wanda."""
import argparse
from pathlib import Path
import torch

from common import fresh_output, load_model, write_json
from matrix import check_linear, prune_mlp


def shrink(model, width):
    config = getattr(model.config, 'text_config', model.config)
    old_width = getattr(config, 'intermediate_size', None)
    if not isinstance(old_width, int) or not 0 < width < old_width:
        raise ValueError('Require a smaller positive intermediate_size')
    mlps = [(n, m) for n, m in model.named_modules() if n.endswith('.mlp') and
            all(hasattr(m, k) for k in ('gate_proj', 'up_proj', 'down_proj')) and 'visual' not in n]
    if len(mlps) != getattr(config, 'num_hidden_layers', None):
        raise ValueError('Only one dense gated MLP per text layer is supported; no MoE or unknown layout')
    for _, m in mlps:
        for layer in (m.gate_proj, m.up_proj, m.down_proj):
            check_linear(layer)
        if m.gate_proj.out_features != old_width or m.up_proj.out_features != old_width or m.down_proj.in_features != old_width:
            raise ValueError('MLP widths disagree with model config')
    selected = {}
    for name, m in mlps:
        with torch.no_grad():
            score = (m.gate_proj.weight.float().norm(dim=1) * m.up_proj.weight.float().norm(dim=1)
                     * m.down_proj.weight.float().norm(dim=0))
            keep = score.argsort(descending=True, stable=True)[:width].sort().values
            selected[name] = keep.cpu().tolist()
        prune_mlp(m, keep)
    config.intermediate_size = width
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    a = p.parse_args()
    out = fresh_output(a.output, [a.model])
    model = load_model(a.model, a.device, a.dtype)
    selected = shrink(model, a.width)
    model.save_pretrained(out/'model', safe_serialization=True)
    write_json(out/'channels.json', selected)
    write_json(out/'result.json', dict(method='uniform gated-MLP magnitude baseline', width=a.width,
               weight_bytes=sum(f.stat().st_size for f in (out/'model').glob('*.safetensors')),
               status='Physically reduced floating-point export; reload, PPL, capacity and decode checks required'))


if __name__ == '__main__':
    main()
