"""Shared offline helpers. Runtime model loading is deliberately fail-closed."""
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def fresh_output(path, inputs=()):
    path = Path(path).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if path == source or path in source.parents or source in path.parents:
            raise ValueError('Output and input must be separate directory trees')
    if path.exists():
        raise FileExistsError(f'Choose a fresh output directory: {path}')
    path.mkdir(parents=True)
    return path


def load_model(path, device='cpu', dtype='float32'):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
    path = Path(path)
    if not (path / 'config.json').is_file():
        raise ValueError('Supply a local model directory containing config.json')
    for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ[key] = '1'
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if getattr(config, 'quantization_config', None) or getattr(getattr(config, 'text_config', None), 'quantization_config', None):
        raise ValueError('These examples require floating-point weights, not packed/FP8 checkpoints')
    cls = AutoModelForImageTextToText if getattr(config, 'vision_config', None) is not None else AutoModelForCausalLM
    return cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                              dtype=getattr(torch, dtype), device_map={'': device},
                              attn_implementation='sdpa').eval()


def read_blocks(path):
    """Explicit token blocks: use the task tokenizer and public training data only."""
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty JSON list of token-ID lists')
    for row in rows:
        if not isinstance(row, list) or len(row) < 2 or any(type(x) is not int or x < 0 for x in row):
            raise ValueError('Each block needs at least two nonnegative integer token IDs')
    return rows


def validate_splits(train, validation):
    if {tuple(x) for x in train} & {tuple(x) for x in validation}:
        raise ValueError('Training and validation contain identical blocks')


def model_logits(model, row, device):
    import torch
    ids = torch.tensor([row], device=device)
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, :-1]
