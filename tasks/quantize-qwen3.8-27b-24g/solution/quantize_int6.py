"""Author reference: data-free INT6/group32 RTN, preserving vision; historical 24G size must be revalidated."""
import argparse,json,shutil,struct
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import load_file,save_file
from llmcompressor import model_free_ptq
from compressed_tensors.quantization import QuantizationArgs,QuantizationScheme

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--submission', type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f'Missing source model: {args.model}')
    if args.submission.exists() and any(args.submission.iterdir()):
        raise SystemExit(f'Refusing to overwrite non-empty submission: {args.submission}')
    args.submission.mkdir(parents=True, exist_ok=True)
    out = args.submission / 'model'
    ignore=['lm_head','re:.*embed_tokens.*','re:.*norm.*','re:.*linear_attn\\.conv1d.*','re:.*linear_attn\\.in_proj_a.*','re:.*linear_attn\\.in_proj_b.*','re:.*visual.*','re:.*vision.*','re:.*mtp.*']
    scheme = QuantizationScheme(
        targets=['Linear'],
        weights=QuantizationArgs(num_bits=6, type='int', symmetric=True,
                                strategy='group', group_size=32))
    model_free_ptq(model_stub=str(args.model), save_directory=str(out),
                   scheme=scheme, ignore=ignore, max_workers=1, device='cuda:0')
    dropped=0
    for f in sorted(out.glob('*.safetensors')):
        with safe_open(f,framework='pt') as sf:names=[k for k in sf.keys() if k.startswith(('mtp.','model.mtp.'))]
        if not names:continue
        data=load_file(f)
        for k in names:del data[k]
        dropped+=len(names)
        if data:
            tmp=f.with_suffix('.new');save_file(data,tmp,metadata={'format':'pt'});tmp.replace(f)
        else:f.unlink()
    weight_map={};total=0
    for f in sorted(out.glob('*.safetensors')):
        with f.open('rb') as h:header=json.loads(h.read(struct.unpack('<Q',h.read(8))[0]))
        for k,v in header.items():
            if k=='__metadata__':continue
            assert k not in weight_map;weight_map[k]=f.name;total+=v['data_offsets'][1]-v['data_offsets'][0]
    (out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':weight_map},indent=2)+'\n')
    c=json.loads((out/'config.json').read_text());c['text_config']['mtp_num_hidden_layers']=0
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    shutil.copyfile(Path(__file__).with_name('inference.py'), args.submission / 'inference.py')
    result=dict(bits=6,group=32,allocation=None,bf16_layers=[],dropped_mtp_tensors=dropped,weight_bytes=total)
    (args.submission / 'recipe.json').write_text(json.dumps(result,indent=2)+'\n');print('BUILD COMPLETE',result,flush=True)
if __name__=='__main__':main()
