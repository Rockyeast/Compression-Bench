"""Offline numerical smoke of pinned author code, not task quality/performance."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tempfile


def author_path(name):
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    manifest = Path(__file__).resolve().parents[1] / 'references' / 'code.json'
    repos = json.loads(manifest.read_text())['repositories']
    entry = next(r for r in repos if r['name'] == name)
    sys.path.insert(0, str((manifest.parent / entry['directory_path']).resolve()))


def svdllm():
    author_path('svd-llm')
    import torch
    import SVDLLM
    from transformers import LlamaConfig, LlamaForCausalLM
    from safetensors.torch import save_file, load_file
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=32)).eval()
    before = sum(p.numel() for p in model.parameters())
    # Identity profile tests the author's decomposition/replacement; it does not
    # estimate real calibration statistics or reproduce paper quality results.
    profile = {0: {n: torch.eye(m.in_features) for n, m in
                   SVDLLM.find_layers(model.model.layers[0]).items()}}
    SVDLLM.whitening('llama', model, profile, 0.5, 'cpu')
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    assert torch.isfinite(expected).all()
    after = sum(p.numel() for p in model.parameters())
    assert after < before
    restored = copy.deepcopy(model)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / 'weights.safetensors')
        save_file({n: t.contiguous().clone() for n, t in model.state_dict().items()}, path)
        restored.load_state_dict(load_file(path))
    with torch.no_grad():
        torch.testing.assert_close(restored(ids, use_cache=False).logits, expected,
                                   atol=0, rtol=0)
    return dict(tool='svdllm', device='cpu', parameters_before=before,
                parameters_after=after, author_whitening=True,
                same_architecture_weight_roundtrip=True, qwen_adapter=False)


def sinq(device):
    author_path('sinq')
    import torch
    from sinq.sinqlinear import SINQLinear, BaseQuantizeConfig
    from sinq.patch_model import AutoSINQHFModel  # Check export/loader imports too.
    torch.manual_seed(42)
    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    linear = torch.nn.Linear(128, 64, bias=False).to(device=device, dtype=dtype)
    quant = SINQLinear(linear, BaseQuantizeConfig(nbits=4, group_size=64,
                      tiling_mode='1D', method='sinq'), compute_dtype=dtype, device=device)
    if device.startswith('cuda') and not quant.use_gemlite:
        raise RuntimeError('CUDA smoke requires GemLite; refusing a silent PyTorch fallback')
    x = torch.randn(3, 128, device=device, dtype=dtype)
    with torch.no_grad():
        y = quant(x)
        assert torch.isfinite(y).all() and y.shape == (3, 64)
        torch.testing.assert_close(y, torch.nn.functional.linear(x, quant.dequantize().to(device=x.device, dtype=x.dtype)),
                                   atol=0.02 if device.startswith('cuda') else 1e-5,
                                   rtol=0.02 if device.startswith('cuda') else 1e-5)
        restored = SINQLinear(None, compute_dtype=dtype, device=device)
        restored.load_state_dict(copy.deepcopy(quant.state_dict()))
        torch.testing.assert_close(restored(x), y)
    assert quant.W_q.dtype == torch.uint8
    assert quant.W_q.numel() * quant.W_q.element_size() == 128 * 64 // 2
    return dict(tool='sinq', device=device, backend='gemlite' if quant.use_gemlite else 'pytorch',
                int4_packed_weight_bytes=quant.W_q.numel(), state_roundtrip=True,
                qwen_adapter=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tool', choices=['svdllm', 'sinq'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    if args.tool == 'svdllm' and args.device != 'cpu':
        parser.error('The SVD-LLM smoke uses a tiny CPU fixture')
    print(json.dumps(svdllm() if args.tool == 'svdllm' else sinq(args.device)))


if __name__ == '__main__':
    main()
