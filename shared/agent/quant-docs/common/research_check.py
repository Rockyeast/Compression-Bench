"""Offline component checks, not a full paper reproduction or packed-model score."""
import argparse
import json
from author_check import author_path


def check(name):
    author_path(name)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(42)
    if name == 'd2quant':
        from d2quant.quantization import DualScaleQuantizer2Bit, DualScaleQuantizer3Bit
        from d2quant.quantization.dac import apply_mean_shift
        from d2quant.quantization import pipeline
        from d2quant.rotation import transforms
        w = torch.randn(16, 128)
        for bits, cls in [(2, DualScaleQuantizer2Bit), (3, DualScaleQuantizer3Bit)]:
            q = cls(); q.configure(bits=bits, num_iters=2); q.find_params(w)
            result = q.quantize(w)
            assert result.shape == w.shape and torch.isfinite(result).all()
        norm = torch.nn.RMSNorm(8)
        x = torch.randn(16, 8)
        apply_mean_shift(norm, [x], [x + .2])
        torch.testing.assert_close(norm.bias, torch.full((8,), .2))
        return {'dual_scale_2_3_bit': True, 'mean_shift': True, 'packed_export': False}
    if name == 'spinquant':
        from train_utils.quant_linear import QuantizeLinear
        from train_utils.optimizer import SGDG
        from train_utils.modeling_llama_quant import LlamaForCausalLM
        from utils import quant_utils
        layer = QuantizeLinear(16, 16, bias=False)
        rotation = torch.nn.Parameter(torch.eye(16))
        optimizer = SGDG([{'params': [rotation], 'stiefel': True}], lr=.01)
        x = torch.randn(4, 16)
        loss = layer(x, R1=rotation).square().mean()
        loss.backward(); optimizer.step()
        assert torch.isfinite(rotation).all() and rotation.grad is not None
        return {'rotation_forward_backward_update': True, 'full_training': False}
    if name == 'qtip':
        from lib.codebook.bitshift import bitshift_codebook, decode_1mad
        from lib.linear.quantized_linear import QuantizedLinear
        values = decode_1mad(torch.arange(256))
        assert values.shape == (256,) and torch.isfinite(values).all()
        return {'codebook_and_linear_import': True, 'trellis_value_decode': True,
                'cuda_inference_kernel': False}
    raise ValueError(name)


def check_gpu(name):
    author_path(name)
    import torch
    import math
    if name != 'qtip':
        from fast_hadamard_transform import hadamard_transform
        x = torch.randn(4, 128, device='cuda', dtype=torch.float16)
        y = hadamard_transform(x, scale=1/math.sqrt(128))
        z = hadamard_transform(y, scale=1/math.sqrt(128))
        torch.testing.assert_close(z, x, atol=.005, rtol=.005)
        return {'hadamard_gpu_roundtrip': True, 'gpu': torch.cuda.get_device_name()}
    import qtip_kernels
    from lib.codebook.bitshift import quantlut_sym
    from lib.utils.kernel_decompress import decode_compressed
    torch.manual_seed(42)
    m=k=256
    compressed=torch.randint(-(2**31),2**31-1,(2*m*k//32,),dtype=torch.int32,device='cuda')
    lookup=torch.randn(512,2,dtype=torch.float16,device='cuda')
    x=torch.randn(k,1,dtype=torch.float16,device='cuda')
    out=torch.zeros(m,1,dtype=torch.float32,device='cuda')
    qtip_kernels.decompress_matvec_16_9_2_1_256_1_256(out,compressed,x,lookup.flatten())
    # Check with author's independent tensor decoder without JIT compilation overhead.
    decoder=getattr(decode_compressed,'_torchdynamo_orig_callable',decode_compressed)
    weight=decoder(16,9,2,1,m,k,compressed,quantlut_sym(lookup,16,9))
    ref=weight.float()@x.float()
    torch.testing.assert_close(out,ref,atol=.03,rtol=.01)
    torch.cuda.synchronize()
    return {'qtip_gpu_random_matvec': True, 'shape': [m,k], 'bits': 2,
            'max_abs_error': (out-ref).abs().max().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tool', choices=['spinquant','d2quant','qtip'])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    result = check(args.tool) if args.device == 'cpu' else check_gpu(args.tool)
    print(json.dumps({'tool': args.tool, 'device': args.device, **result}))


if __name__ == '__main__':
    main()
