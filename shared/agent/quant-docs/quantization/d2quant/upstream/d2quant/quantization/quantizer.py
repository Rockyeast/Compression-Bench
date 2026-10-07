import torch


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2 ** (bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(round_ste(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


# Pack the int tensor. Each uint8 stores two int4 value.

# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.


class WeightQuantizer(torch.nn.Module):
    """From GPTQ Repo"""

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2 ** (bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x, hinv1=None):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        # --- prepare d_vec (1, N) if provided ---
        d_vec = None
        if hinv1 is not None:
            if hinv1.dim() == 2:
                d = torch.diag(hinv1)
            else:
                d = hinv1
            d = d.to(dev).float()
            N = x.shape[1]
            d = d[:N]
            eps = 1e-6
            d_vec = d.view(1, -1).clamp(min=eps)  # [1, N]

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(
                        x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
                    )

                # q -= x

                diff = q - x
                if d_vec is not None:
                    diff = diff / d_vec  # broadcast over C

                diff.abs_()
                diff.pow_(self.norm)
                err = torch.sum(diff, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def forward(self, x):
        return self.quantize(x)

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def find_qlayers(module, layers=(torch.nn.Linear,), name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_qlayers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


# class Quant3Linear(torch.nn.Module):

#     def __init__(self, infeatures, outfeatures, faster=False):
#         super().__init__()
#         self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
#         self.register_buffer('scales', torch.zeros((outfeatures, 1)))
#         self.register_buffer('bias', torch.zeros(outfeatures))
#         self.register_buffer(
#             'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
#         )
#         self.faster = faster

#     def pack(self, linear, scales, zeros):
#         self.zeros = zeros * scales
#         self.scales = scales.clone()
#         if linear.bias is not None:
#             self.bias = linear.bias.clone()

#         intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
#         intweight = intweight.t().contiguous()
#         intweight = intweight.numpy().astype(np.uint32)
#         qweight = np.zeros(
#             (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
#         )
#         i = 0
#         row = 0
#         while row < qweight.shape[0]:
#             for j in range(i, i + 10):
#                 qweight[row] |= intweight[j] << (3 * (j - i))
#             i += 10
#             qweight[row] |= intweight[i] << 30
#             row += 1
#             qweight[row] |= (intweight[i] >> 2) & 1
#             i += 1
#             for j in range(i, i + 10):
#                 qweight[row] |= intweight[j] << (3 * (j - i) + 1)
#             i += 10
#             qweight[row] |= intweight[i] << 31
#             row += 1
#             qweight[row] |= (intweight[i] >> 1) & 0x3
#             i += 1
#             for j in range(i, i + 10):
#                 qweight[row] |= intweight[j] << (3 * (j - i) + 2)
#             i += 10
#             row += 1

#         qweight = qweight.astype(np.int32)
#         self.qweight = torch.from_numpy(qweight)

#     def forward(self, x):
#         if x.shape[-1] == x.numel():
#             outshape = list(x.shape)
#             y = self.bias.clone()
#             outshape[-1] = self.bias.numel()
#             dtype = x.dtype
#             if self.faster:
#                 x = x.half()
#                 quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
#             else:
#                 x = x.float()
#                 quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
#             y = y.to(dtype)
#             return y.reshape(outshape)
#         raise ValueError('Only supports a single token currently.')

# def make_quant3(module, names, name='', faster=False):
#     if isinstance(module, Quant3Linear):
#         return
#     for attr in dir(module):
#         tmp = getattr(module, attr)
#         name1 = name + '.' + attr if name != '' else attr
#         if name1 in names:
#             setattr(
#                 module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
#             )
#     for name1, child in module.named_children():
#         make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)
