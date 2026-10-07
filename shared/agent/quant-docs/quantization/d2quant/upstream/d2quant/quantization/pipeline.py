import math
import time
import tqdm
import torch
import torch.nn as nn

from d2quant import config
from d2quant import model_utils
from . import dac
from . import diagnostics
from . import dsq_2bit
from . import dsq_3bit
from . import quantizer as base_quantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTAQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.dXXT = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.fp_inp = []
        # self.inp = []
        self.disable_col_comp = False
        self.fp_inp_bias = []

    def add_batch(self, inp, out):

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        # self.inp.append(inp)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.dXXT *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        dX = self.fp_inp[0].float() * math.sqrt(2 / self.nsamples) - inp
        self.dXXT += dX.matmul(inp.t())

        del self.fp_inp[0]

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=128,
        dsq=False,
        logger=None,
    ):
        """Quantize one linear layer with GPTAQ and optional DSQ."""
        start_time = time.time()
        weights = self.layer.weight.data.float().clone()

        if not self.quantizer.ready():
            self.quantizer.find_params(weights)

        hessian = self.H
        del self.H
        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weights[:, dead] = 0
        self.dXXT[:, dead] = 0

        from .gar import (
            compose_final_perm,
            compute_global_perm,
            compute_local_perms,
            invert_perm,
        )

        local_perms = compute_local_perms(torch.diag(hessian), blocksize)
        global_perm = compute_global_perm(torch.diag(hessian), blocksize)
        permutation = compose_final_perm(local_perms, global_perm, blocksize)
        weights = weights[:, permutation]
        hessian = hessian[permutation][:, permutation]
        self.dXXT = self.dXXT[permutation][:, permutation]

        quantized = torch.zeros_like(weights)
        damp = percdamp * torch.mean(torch.diag(hessian))
        diagonal = torch.arange(self.columns, device=self.dev)
        hessian[diagonal, diagonal] += damp
        hessian_inverse = torch.linalg.cholesky(hessian)
        hessian_inverse = torch.cholesky_inverse(hessian_inverse)
        hessian_inverse = torch.linalg.cholesky(hessian_inverse, upper=True)

        # GPTAQ asymmetric error compensation.
        compensation = (
            0.25 * ((self.dXXT @ hessian_inverse.T).triu_(diagonal=1)) @ hessian_inverse
        )
        del self.dXXT

        for block_start in range(0, self.columns, blocksize):
            block_end = min(block_start + blocksize, self.columns)
            count = block_end - block_start
            block = weights[:, block_start:block_end].clone()
            quantized_block = torch.zeros_like(block)
            errors = torch.zeros_like(block)
            inverse_block = hessian_inverse[
                block_start:block_end, block_start:block_end
            ]
            compensation_block = compensation[
                block_start:block_end, block_start:block_end
            ]

            for column in range(count):
                weight = block[:, column]
                diagonal_value = inverse_block[column, column]

                if groupsize != -1 and (block_start + column) % groupsize == 0:
                    group = weights[
                        :, (block_start + column) : (block_start + column + groupsize)
                    ]
                    self.quantizer.find_params(group)

                if dsq:
                    qweight = self.quantizer.quantize(
                        weight.unsqueeze(1), column
                    ).flatten()
                else:
                    qweight = self.quantizer.quantize(weight.unsqueeze(1)).flatten()

                quantized_block[:, column] = qweight
                error = (weight - qweight) / diagonal_value
                block[:, column:] -= error.unsqueeze(1) @ inverse_block[
                    column, column:
                ].unsqueeze(0) - weight.unsqueeze(1) @ compensation_block[
                    column, column:
                ].unsqueeze(0)
                errors[:, column] = error

            quantized[:, block_start:block_end] = quantized_block
            weights[:, block_end:] -= (
                errors @ hessian_inverse[block_start:block_end, block_end:]
                - block @ compensation[block_start:block_end, block_end:]
            )

        torch.cuda.synchronize()
        quantized = quantized[:, invert_perm(permutation)]
        self.layer.weight.data = quantized.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if torch.isnan(self.layer.weight.data).any():
            raise ValueError("NaN detected in quantized weights")
        if logger is not None:
            logger.debug("Quantized layer in %.2fs", time.time() - start_time)

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        torch.cuda.empty_cache()
        config.cleanup_memory(verbose=False)


@torch.no_grad()
def quantize_model(model, dataloader, dev, args, logger):
    """Quantize a model layer-by-layer with GPTAQ, DSQ, and optional DAC."""
    logger.info("Quantizing %d transformer layers", len(model.model.layers))

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    # model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module, attention_type=None):
            super().__init__()
            self.module = module
            self.attention_type = attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)

            raise ValueError

    if "qwen" in args.model.lower():
        attention_type = layers[0].attention_type
    else:
        attention_type = None

    layers[0] = Catcher(layers[0], attention_type=attention_type)
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}

    sequential = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]

    fp_inputs_cache = model_utils.FPInputsCache(sequential)
    fp_inps = inps.clone()

    if args.dac:
        semi_fp_inps = inps.clone()

    if args.dac:
        postln_names = ["post_attention_layernorm"]
        ln_cache_fp = model_utils.LnOutCache(postln_names)

    for i in tqdm.trange(len(layers), desc="D²Quant", unit="layer"):
        layer = layers[i].to(dev)
        full = base_quantizer.find_qlayers(layer, layers=[torch.nn.Linear])

        fp_inputs_cache.add_hook(full)

        for j in range(args.nsamples):
            fp_inps[j] = layer(
                fp_inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]
        fp_inputs_cache.clear_hook()
        if args.dac:
            layernorms = {n: getattr(layer, n) for n in postln_names}
            ln_cache_fp.add_hook(layernorms)
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            ln_cache_fp.clear_hook()

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                layer_weight_bits = args.w_bits
                layer_weight_sym = not args.w_asym
                gptq[name] = GPTAQ(subset[name])
                gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

                if args.dsq and "down_proj" in name:
                    if args.w_bits == 2:
                        gptq[name].quantizer = dsq_2bit.DualScaleQuantizer()
                        gptq[name].quantizer.configure(
                            bits=2,
                            num_iters=args.num_iters,
                        )
                    elif args.w_bits == 3:
                        gptq[name].quantizer = dsq_3bit.DualScaleQuantizer()
                        gptq[name].quantizer.configure(
                            bits=3,
                            num_iters=args.num_iters,
                        )
                else:
                    gptq[name].quantizer = base_quantizer.WeightQuantizer()
                    gptq[name].quantizer.configure(
                        layer_weight_bits,
                        perchannel=True,
                        sym=layer_weight_sym,
                        mse=args.w_clip,
                    )

            def add_batch(name, layer_quantizers=gptq):
                def tmp(_, inp, out):
                    layer_quantizers[name].add_batch(inp[0].data, out.data)

                return tmp

            first_module_name = list(subset.keys())[0]

            if args.dac:
                if first_module_name == "mlp.up_proj":
                    y_teacher_list = ln_cache_fp.cache["post_attention_layernorm"]

                    y_student_list = []

                    def _cap_inp(_m, inp, out):
                        x = inp[0].detach()  # [B,S,H] or [S,H]
                        if x.dim() == 3:
                            x = x.reshape(-1, x.shape[-1])  # [T,H]
                        y_student_list.append(x)

                    h_cap = subset[first_module_name].register_forward_hook(_cap_inp)
                    for j in range(args.nsamples):
                        outs[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )[0]
                    h_cap.remove()

                    diagnostics.log_snr_from_lists(
                        y_student_list, y_teacher_list, logger
                    )

                    layernorm = layer.post_attention_layernorm
                    ln_tag = "post_attention_layernorm"

                    bias = dac.apply_deviation_aware_correction(
                        layernorm,
                        y_student_list,
                        y_teacher_list,
                        clip=10.0,
                        lmbd=1.0,
                        # snr_th=2.0,
                        # keep_ratio=0.25,
                        # explain_th=0.0001,
                    )

                    if isinstance(bias, torch.Tensor):
                        if bias.dim() == 1:
                            bias_2d = bias.view(1, -1)
                        else:
                            bias_2d = bias

                    n_pairs = min(len(y_student_list), len(y_teacher_list))
                    mse_before_sum = 0.0
                    mse_after_sum = 0.0

                    for t in range(n_pairs):
                        ys = y_student_list[t].float()
                        yt = y_teacher_list[t].float()

                        mse_before_sum += ((ys - yt) ** 2).mean().item()
                        ys_aligned = ys + bias_2d
                        mse_after_sum += ((ys_aligned - yt) ** 2).mean().item()

                    mse_before_avg = mse_before_sum / max(n_pairs, 1)
                    mse_after_avg = mse_after_sum / max(n_pairs, 1)

                    logger.debug(
                        "[DAC] layer %d %s: mean MSE before/after over %d samples: %.6e %.6e",
                        i,
                        ln_tag,
                        n_pairs,
                        mse_before_avg,
                        mse_after_avg,
                    )

            if args.dac:
                if first_module_name == "mlp.up_proj":
                    y_student_list.clear()
                    ln_cache_fp.clear_cache()
                    del y_student_list
                    del h_cap

            if args.dac:
                if first_module_name == "mlp.up_proj":
                    for j in range(args.nsamples):
                        semi_fp_inps[j] = layer(
                            inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )[0]

            handle = subset[first_module_name].register_forward_hook(
                add_batch(first_module_name)
            )
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            handle.remove()

            # copy H and dXXT
            for name in subset:
                if name != first_module_name:
                    gptq[name].H = gptq[first_module_name].H
                    gptq[name].dXXT = gptq[first_module_name].dXXT
                    # gptq[name].inp = gptq[first_module_name].inp

            for name in subset:
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    dsq=args.dsq and "down_proj" in name,
                    logger=logger,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        fp_inputs_cache.clear_cache()
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if args.dac:
        y_teacher_list = []
        y_student_list = []

        def _to_2d(x: torch.Tensor) -> torch.Tensor:
            # x: [S,H] or [1,S,H] or [B,S,H]  -> [T,H]
            if x.dim() == 2:
                return x
            if x.dim() == 3:
                return x.reshape(-1, x.shape[-1])
            return x.view(-1, x.shape[-1])

        layernorm = model.model.norm
        layernorm = layernorm.to(dev)
        ln_tag = "last_layernorm"

        # semi_fp_inps: teacher inputs; inps: student inputs
        for j in range(args.nsamples):
            # teacher
            xt = semi_fp_inps[j]
            if xt.dim() == 2:
                xt = xt.unsqueeze(0)  # -> [1,S,H]
            yt = layernorm(xt).detach()
            if yt.dim() == 3 and yt.shape[0] == 1:
                yt = yt.squeeze(0)  # -> [S,H]
            y_teacher_list.append(_to_2d(yt))

            # student
            xs = inps[j]
            if xs.dim() == 2:
                xs = xs.unsqueeze(0)
            ys = layernorm(xs).detach()
            if ys.dim() == 3 and ys.shape[0] == 1:
                ys = ys.squeeze(0)
            y_student_list.append(_to_2d(ys))

        diagnostics.log_snr_from_lists(y_student_list, y_teacher_list, logger)
        bias = dac.apply_mean_shift(
            layernorm,
            y_student_list,
            y_teacher_list,
            clip=10.0,
            lmbd=1.0,
        )

        if isinstance(bias, torch.Tensor):
            if bias.dim() == 1:
                bias_2d = bias.view(1, -1)
            else:
                bias_2d = bias

        n_pairs = min(len(y_student_list), len(y_teacher_list))
        mse_before_sum = 0.0
        mse_after_sum = 0.0

        for t in range(n_pairs):
            ys = y_student_list[t].float()
            yt = y_teacher_list[t].float()

            mse_before_sum += ((ys - yt) ** 2).mean().item()
            ys_aligned = ys + bias_2d
            mse_after_sum += ((ys_aligned - yt) ** 2).mean().item()

        mse_before_avg = mse_before_sum / max(n_pairs, 1)
        mse_after_avg = mse_after_sum / max(n_pairs, 1)

        logger.debug(
            "[DAC] layer %d %s: mean MSE before/after over %d samples: %.6e %.6e",
            i,
            ln_tag,
            n_pairs,
            mse_before_avg,
            mse_after_avg,
        )

        y_student_list.clear()
        y_teacher_list.clear()
        layernorm = layernorm.to("cpu")
        del y_student_list
        del y_teacher_list

    model.config.use_cache = use_cache
    config.cleanup_memory(verbose=True)
    logger.info("Quantization complete")

    return quantizers
