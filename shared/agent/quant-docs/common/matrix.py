"""PyTorch baselines for physical gated-MLP pruning and truncated-SVD linear layers.

These are small reusable implementations, not Wanda or SVD-LLM reproductions.
Use with floating-point nn.Linear modules; packed quantized tensors are rejected.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from common import fresh_output, write_json


def check_linear(layer):
    if type(layer) is not nn.Linear or layer.weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError('Expected a plain floating-point nn.Linear')


def prune_mlp(mlp, keep):
    """Physically slice matching gate/up rows and down columns. Update config separately."""
    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    for layer in (gate, up, down):
        check_linear(layer)
    width = gate.out_features
    if up.out_features != width or down.in_features != width or gate.in_features != up.in_features:
        raise ValueError('Incompatible gated-MLP shapes')
    keep = torch.as_tensor(keep, dtype=torch.long, device=gate.weight.device)
    if keep.ndim != 1 or not 0 < len(keep) < width or len(keep.unique()) != len(keep) or int(keep.min()) < 0 or int(keep.max()) >= width:
        raise ValueError('Require unique valid channels and a strictly smaller width')
    for name, old, rows in [('gate_proj', gate, True), ('up_proj', up, True), ('down_proj', down, False)]:
        new = nn.Linear(old.in_features if rows else len(keep), len(keep) if rows else old.out_features,
                        bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight.index_select(0 if rows else 1, keep.to(old.weight.device)))
            if old.bias is not None:
                new.bias.copy_(old.bias.index_select(0, keep.to(old.bias.device)) if rows else old.bias)
        setattr(mlp, name, new)
    if hasattr(mlp, 'intermediate_size'):
        mlp.intermediate_size = len(keep)
    return mlp


class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False, *, device=None, dtype=None):
        super().__init__()
        self.right = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.left = nn.Linear(rank, out_features, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        return self.left(self.right(x))

    @classmethod
    def from_linear(cls, layer, rank):
        check_linear(layer)
        n, m = layer.weight.shape
        if not 0 < rank <= min(n, m) or rank*(n+m) >= n*m:
            raise ValueError('Rank must physically reduce the matrix parameter count')
        u, s, vh = torch.linalg.svd(layer.weight.detach().float(), full_matrices=False)
        result = cls(m, n, rank, layer.bias is not None, device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            result.left.weight.copy_(u[:, :rank] * s[:rank])
            result.right.weight.copy_(vh[:rank])
            if layer.bias is not None:
                result.left.bias.copy_(layer.bias)
        return result

    def save(self, directory):
        directory = fresh_output(directory)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}, str(directory/'factors.safetensors'))
        write_json(directory/'factors.json', dict(in_features=self.right.in_features,
                   out_features=self.left.out_features, rank=self.right.out_features,
                   bias=self.left.bias is not None))

    @classmethod
    def load(cls, directory, device='cpu'):
        directory = Path(directory)
        state = load_file(str(directory/'factors.safetensors'), device=device)
        result = cls(**json.loads((directory/'factors.json').read_text()), device=device,
                     dtype=state['left.weight'].dtype)
        result.load_state_dict(state, strict=True)
        return result


def main():
    p = argparse.ArgumentParser(description='Factor one floating-point matrix; not a complete model export.')
    p.add_argument('--weights', type=Path, required=True, help='SafeTensors file')
    p.add_argument('--key', required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from safetensors import safe_open
    with safe_open(a.weights, framework='pt', device='cpu') as f:
        w = f.get_tensor(a.key)
    if w.ndim != 2:
        raise ValueError('Expected a matrix')
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False, dtype=w.dtype)
    layer.weight = nn.Parameter(w, requires_grad=False)
    FactorizedLinear.from_linear(layer, a.rank).save(a.output)


if __name__ == '__main__':
    main()
