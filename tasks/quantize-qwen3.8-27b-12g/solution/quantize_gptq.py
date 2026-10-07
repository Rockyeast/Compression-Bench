"""Fixed author GPTQ candidate: same bit allocation as the RTN experiment."""
import argparse
import json
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

IGNORE = ['re:.*embed_tokens.*', 're:.*norm.*',
          r're:.*linear_attn\.conv1d.*', r're:.*linear_attn\.in_proj_a.*',
          r're:.*linear_attn\.in_proj_b.*', 're:.*visual.*', 're:.*vision.*',
          r're:^mtp\..*']
OPTIONAL = ('mtp.',)  # Text-only auxiliary module; visual weights are protected.

BODY_TARGET = (r're:.*\.(?:q_proj|k_proj|v_proj|o_proj|in_proj_qkv|'
               r'in_proj_z|out_proj|gate_proj|up_proj|down_proj)$')
TRAIN_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    args = parser.parse_args()
    assert version('llmcompressor') == '0.13.0'
    assert version('compressed-tensors') == '0.18.0'
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from compressed_tensors.quantization import preset_name_to_scheme
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    torch.set_num_threads(8)
    torch.manual_seed(42)
    start = time.monotonic()
    submission = args.run / 'submission'
    submission.mkdir()
    output = submission / 'model'
    rows = pq.read_table(args.calibration, columns=['text'])['text'].to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                               trust_remote_code=False)
    tokens = tokenizer('\n\n'.join(rows), add_special_tokens=False).input_ids
    # Disjoint fixed-length blocks from TRAIN only, sampled without replacement.
    starts = random.Random(42).sample(list(range(0, len(tokens) - 2048 + 1, 2048)), 128)
    samples = [{'input_ids': torch.tensor(tokens[s:s + 2048], dtype=torch.long),
                'attention_mask': torch.ones(2048, dtype=torch.long)} for s in starts]
    loader = DataLoader(samples, batch_size=4, shuffle=False)
    metadata = {
        'algorithm': 'unmodified LLM Compressor GPTQModifier',
        'llmcompressor': version('llmcompressor'), 'compressed_tensors': version('compressed-tensors'),
        'body_bits': 3, 'head_bits': 4, 'group_size': 128, 'actorder': None,
        'dampening_frac': 0.01, 'block_size': 128, 'hessian_precision': 'stock float32',
        'pipeline': 'sequential', 'sequential_offload_device': 'cpu',
        'dataset': 'Salesforce/wikitext', 'subset': 'wikitext-2-raw-v1', 'split': 'train',
        'dataset_revision': TRAIN_REVISION,
        'seed': 42, 'num_calibration_samples': 128, 'max_seq_length': 2048,
        'batch_size': 4, 'calibration_token_count': 128 * 2048, 'start_offsets': starts,
        'ignore': IGNORE,
    }
    (args.run / 'gptq-recipe.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Calibration fixed: 128 x 2048 TRAIN tokens, seed 42; loading original BF16', flush=True)
    del tokens, rows
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True, trust_remote_code=False)
    groups = {'body_int3': preset_name_to_scheme('W3A16', targets=[BODY_TARGET]),
              'head_int4': preset_name_to_scheme('W4A16', targets=['re:.*lm_head$'])}
    recipe = GPTQModifier(config_groups=groups, ignore=IGNORE,
                          actorder=None, dampening_frac=0.01, block_size=128,
                          offload_hessians=False)
    print('Starting stock GPTQ sequential calibration', flush=True)
    oneshot(model=model, tokenizer=tokenizer, dataset=loader, recipe=recipe,
            num_calibration_samples=128, max_seq_length=2048, batch_size=4,
            shuffle_calibration_samples=False, pipeline='sequential',
            sequential_targets=['Qwen3_5DecoderLayer'], sequential_offload_device='cpu',
            output_dir=str(output), tie_word_embeddings=False, save_compressed=True)
    del model

    # Reuse the original architecture/config and names. Remove only optional
    # MTP tensors from this new submission; preserve all vision tensors.
    index_path = output / 'model.safetensors.index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {'metadata': {}, 'weight_map': {}}
    weight_map, total_size, removed, packed = {}, 0, [], 0
    for shard in sorted(output.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as handle:
            dropped = [n for n in handle.keys() if n.startswith(OPTIONAL)]
        if dropped:
            tensors = load_file(shard)
            for name in dropped:
                del tensors[name]
            removed.extend(dropped)
            if not tensors:
                shard.unlink()
                continue
            temporary = shard.with_suffix('.tmp')
            save_file(tensors, temporary, metadata={'format': 'pt'})
            temporary.replace(shard)
            del tensors
        with safe_open(shard, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                assert name not in weight_map
                weight_map[name] = shard.name
                packed += name.endswith('.weight_packed')
        import struct
        with shard.open('rb') as stream:
            header = json.loads(stream.read(struct.unpack('<Q', stream.read(8))[0]))
        total_size += sum(v['data_offsets'][1] - v['data_offsets'][0]
                          for k, v in header.items() if k != '__metadata__')
    assert packed == 401, f'Expected 400 INT3 body + one INT4 head; got {packed}'
    index.update(metadata={'total_size': total_size}, weight_map=weight_map)
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    config = json.loads((output / 'config.json').read_text())
    qconfig = config['quantization_config']
    bit_counts = {}
    for group in qconfig['config_groups'].values():
        bits = group['weights']['num_bits']
        bit_counts[bits] = bit_counts.get(bits, 0) + 1
        assert group['weights'].get('actorder') is None
        # Exported gate/up targets must also match vLLM's fused gate_up_proj.
        group['targets'] = ([r're:.*(?:self_attn|linear_attn|mlp)\..*']
                            if bits == 3 else ['re:.*lm_head$'])
    assert bit_counts == {3: 1, 4: 1}, bit_counts
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    shutil.copyfile(args.inference, submission / 'inference.py')
    (submission / 'inference.py').chmod(0o755)
    metadata.update(quantization_seconds=time.monotonic() - start,
                    packed_text_weights=packed, removed_optional_tensors=removed)
    (args.run / 'quantization.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'GPTQ DONE: {packed} packed matrices; {metadata["quantization_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
