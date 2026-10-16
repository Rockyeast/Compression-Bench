"""Offline SINQ author API entry; exported runtime is not a stock GPTQ/vLLM model."""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'common'))

from author_check import author_path
from common import fresh_output, load_model, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--bits', type=int, choices=[2, 3, 4, 5, 6, 8], default=4)
    p.add_argument('--group-size', type=int, choices=[32, 64, 128], default=64)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    args = p.parse_args()
    author_path('sinq')
    import torch
    from transformers import AutoTokenizer
    from sinq.patch_model import AutoSINQHFModel
    from sinq.sinqlinear import SINQLinear, BaseQuantizeConfig
    if not (args.model / 'config.json').is_file():
        p.error('--model must be a local model directory')
    if args.device == 'cuda':
        # Fail early with the actual error, rather than silently use slow fallback.
        import gemlite
    model = load_model(args.model, device='cpu', dtype=args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                               trust_remote_code=False)
    output = fresh_output(args.output, inputs=[args.model])
    config = BaseQuantizeConfig(nbits=args.bits, group_size=args.group_size,
                               tiling_mode='1D', method='sinq')
    AutoSINQHFModel.quantize_model(model, tokenizer=tokenizer, quant_config=config,
                                  compute_dtype=getattr(torch, args.dtype), device=args.device,
                                  use_unpack_kernel=False)
    modules = [(n, m) for n, m in model.named_modules() if isinstance(m, SINQLinear)]
    if not modules:
        raise RuntimeError('Author model walker did not quantize any linear layers')
    AutoSINQHFModel.save_quantized_safetensors(model, tokenizer, str(output / 'model'),
                                              max_shard_size='4GB')
    write_json(output / 'report.json', dict(method='sinq', bits=args.bits,
        group_size=args.group_size, quantized_modules=[n for n, _ in modules],
        gemlite_modules=[n for n, m in modules if m.use_gemlite],
        runtime='SINQ author loader; not stock vLLM/GPTQ',
        task_quality_and_speed_validated=False))


if __name__ == '__main__':
    main()
