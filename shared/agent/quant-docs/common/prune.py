"""Offline Wanda/SparseGPT through pinned LLM Compressor. Dense-zero export only."""
import argparse
from importlib.metadata import version
from pathlib import Path
import time

from common import fresh_output, load_model, read_blocks, write_json


def recipe(method, targets, sparsity, structure):
    from llmcompressor.modifiers.pruning import SparseGPTModifier, WandaPruningModifier
    if not 0 < sparsity < 1 or structure not in ('0:0', '2:4'):
        raise ValueError('Require 0 < sparsity < 1 and mask 0:0 or 2:4')
    if structure == '2:4' and sparsity != 0.5:
        raise ValueError('2:4 requires sparsity 0.5')
    cls = {'wanda': WandaPruningModifier, 'sparsegpt': SparseGPTModifier}[method]
    return cls(targets=targets, sparsity=sparsity, mask_structure=structure,
               ignore=['re:.*lm_head', 're:.*visual.*'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--calibration', required=True, type=Path, help='JSON token blocks')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--method', choices=['wanda', 'sparsegpt'], default='wanda')
    p.add_argument('--targets', nargs='+', required=True, help='Explicit module names or re: patterns')
    p.add_argument('--sparsity', type=float, default=0.5)
    p.add_argument('--mask', choices=['0:0', '2:4'], default='0:0')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    a = p.parse_args()
    if version('llmcompressor') != '0.13.0':
        raise RuntimeError('Use /opt/quantize/bin/python with llmcompressor==0.13.0')
    modifier = recipe(a.method, a.targets, a.sparsity, a.mask)
    blocks = read_blocks(a.calibration)
    out = fresh_output(a.output, [a.model, a.calibration])
    from datasets import Dataset
    from llmcompressor import oneshot
    model = load_model(a.model, a.device, a.dtype)
    started = time.monotonic()
    oneshot(model=model, recipe=modifier,
            dataset=Dataset.from_dict({'input_ids': blocks, 'attention_mask': [[1]*len(x) for x in blocks]}),
            num_calibration_samples=len(blocks), max_seq_length=max(map(len, blocks)),
            batch_size=1, pipeline='basic', shuffle_calibration_samples=False)
    model.save_pretrained(out / 'model', safe_serialization=True, save_compressed=False)
    write_json(out / 'result.json', dict(method=a.method, targets=a.targets,
               elapsed_seconds=time.monotonic()-started,
               weight_bytes=sum(f.stat().st_size for f in (out/'model').glob('*.safetensors')),
               status='Dense-zero export; no storage reduction or sparse acceleration claimed. Reload and Checker required.'))


if __name__ == '__main__':
    main()
