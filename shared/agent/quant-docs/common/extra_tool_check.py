"""Tiny offline component checks, not full-model adaptation or quality evaluation."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('WANDB_MODE', 'disabled')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tool', choices=['modelopt', 'torchao', 'hqq', 'quanto', 'bitsandbytes', 'llmc', 'slicegpt', 'sigmascale'])
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = p.parse_args()
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(2)
    device = args.device
    x = torch.randn(4, 128, device=device)
    layer = torch.nn.Linear(128, 64, bias=False).to(device).eval()
    ref = layer(x).detach()
    root = Path(__file__).resolve().parents[1]
    if args.tool == 'modelopt':
        import modelopt.torch.quantization as mtq
        model = torch.nn.Sequential(layer)
        mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(x))
        y = model(x)
    elif args.tool == 'torchao':
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        model = torch.nn.Sequential(layer)
        quantize_(model, Int8WeightOnlyConfig())
        y = model(x)
    elif args.tool == 'hqq':
        from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
        model = HQQLinear(layer, BaseQuantizeConfig(nbits=4, group_size=64), compute_dtype=torch.float32, device=device)
        y = model(x)
    elif args.tool == 'quanto':
        from optimum.quanto import quantize, freeze, qint8
        model = torch.nn.Sequential(layer)
        quantize(model, weights=qint8)
        freeze(model)
        y = model(x)
    elif args.tool == 'bitsandbytes':
        if device != 'cuda':
            raise ValueError('This check exercises the installed CUDA backend; use --device cuda')
        import bitsandbytes as bnb
        model = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.float32, quant_type='nf4')
        model.load_state_dict(layer.cpu().state_dict())
        model = model.cuda()
        y = model(x)
    elif args.tool == 'llmc':
        sys.path.insert(0, str(root / 'quantization/llmc/upstream'))
        from llmc.compression.quantization.quant import IntegerQuantizer
        q = IntegerQuantizer(bit=4, symmetric=True, granularity='per_group', group_size=64)
        w = q.fake_quant_weight_dynamic(layer.weight.detach())
        y = torch.nn.functional.linear(x, w)
    elif args.tool == 'slicegpt':
        from slicegpt import rotate
        from slicegpt.config import config
        config.device = torch.device(device)
        values, vectors = rotate.pca_calc([x.unsqueeze(0)])
        torch.testing.assert_close(vectors.T @ vectors, torch.eye(128, device=vectors.device, dtype=vectors.dtype), atol=1e-5, rtol=1e-5)
        assert torch.isfinite(values).all()
        print('slicegpt PCA: finite eigenvalues and orthogonal basis OK')
        return
    else:
        sys.path.insert(0, str(root / 'other-compression/low-rank/sigmascale/upstream'))
        from utils import low_rank_utils  # Author modules have a circular import; use the author's entry order.
        from modules.linears import SVDLinear
        a = torch.randn(64, 8, device=device)
        b = torch.randn(8, 128, device=device)
        model = SVDLinear(a, b)
        y = model(x)
        ref = torch.nn.functional.linear(x, a @ b)
        torch.testing.assert_close(y, ref, atol=1e-4, rtol=1e-4)
    assert y.shape == ref.shape and torch.isfinite(y).all()
    error = (y-ref).square().mean() / ref.square().mean().clamp_min(1e-12)
    assert error.item() < 0.05, error.item()
    print(f'{args.tool}: finite forward OK, relative MSE={error.item():.6g}, device={device}')


if __name__ == '__main__':
    main()
