import torch
import transformers
import logging
import functools

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
QWEN3_MODEL = transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM
QWEN3_LAYER = transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer
QWEN2_MODEL = transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM
QWEN2_LAYER = transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer


def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, QWEN3_MODEL):
        return QWEN3_MODEL
    elif isinstance(model, QWEN2_MODEL):
        return QWEN2_MODEL
    else:
        raise ValueError(f"Unknown model type {model}")


def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization!
    pass


def skip_initialization():
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip


def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if isinstance(model, OPT_MODEL):
        return model.model.decoder.layers
    if isinstance(model, LLAMA_MODEL):
        return model.model.layers
    if isinstance(model, QWEN3_MODEL):
        return model.model.layers
    if isinstance(model, QWEN2_MODEL):
        return model.model.layers
    raise NotImplementedError


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name, dtype="auto", token=hf_token, low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info(
        "---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen)
    )
    return model


def get_qwen(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype="auto", token=hf_token, low_cpu_mem_usage=True
    )
    model.seqlen = 2048
    logging.info(
        "---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen)
    )
    return model


def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(
        model_name, dtype="auto", low_cpu_mem_usage=True
    )
    model.seqlen = model.config.max_position_embeddings
    logging.info(
        "---> Loading {} Model with seq_len: {}".format(model_name, model.seqlen)
    )
    return model


def get_model(model_name, hf_token=None):
    if "llama" in model_name.lower():
        return get_llama(model_name, hf_token)
    elif "opt" in model_name.lower():
        return get_opt(model_name)
    elif "qwen" in model_name.lower():
        return get_qwen(model_name, hf_token)
    else:
        raise ValueError(f"Unknown model {model_name}")


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, QWEN3_MODEL):
        model_type = QWEN3_MODEL
    elif isinstance(model, QWEN2_MODEL):
        model_type = QWEN2_MODEL
    else:
        raise ValueError(f"Unknown model type {model}")
    return model_type


def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == QWEN3_MODEL or model_type == QWEN2_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model_type == QWEN3_MODEL or model_type == QWEN2_MODEL:
        return [layer for layer in model.model.layers]
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_lm_head(model, model_type):
    if model_type == LLAMA_MODEL:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif (
        model_type == QWEN3_MODEL or model_type == QWEN2_MODEL
    ):  # Qwen3 uses a different lm_head structure
        return model.lm_head
    else:
        raise ValueError(f"Unknown model type {model_type}")


def get_pre_head_layernorm(model, model_type):
    if model_type == LLAMA_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(
            pre_head_layernorm, transformers.models.llama.modeling_llama.LlamaRMSNorm
        )
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == QWEN3_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(
            pre_head_layernorm, transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm
        )
    elif model_type == QWEN2_MODEL:
        pre_head_layernorm = model.model.norm
        assert isinstance(
            pre_head_layernorm, transformers.models.qwen2.modeling_qwen2.Qwen2RMSNorm
        )
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return pre_head_layernorm


def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    elif model_type == QWEN3_MODEL or model_type == QWEN2_MODEL:
        return model.config.intermediate_size
    else:
        raise ValueError(f"Unknown model type {model_type}")


def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class FPInputsCache:
    """
    class for saving the full-precision output in each layer.
    """

    def __init__(self, sequential):
        self.fp_cache = {}
        self.names = sequential[0] + sequential[1] + sequential[2] + sequential[3]
        for name in self.names:
            self.fp_cache[name] = []
        self.handles = []

    def cache_fp_input(self, m, inp, out, name):
        inp = inp[0].detach()
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        self.fp_cache[name] += [inp.t()]

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self.cache_fp_input, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        torch.cuda.empty_cache()

    def clear_cache(self):
        for name in self.names:
            self.fp_cache[name] = []


class LnOutCache:
    """ """

    def __init__(self, names, to_cpu=False):
        # self.names = list(names)
        self.names = names
        self.to_cpu = to_cpu

        self.cache = {name: [] for name in self.names}
        self.handles = []

    def _hook(self, m, inp, out, name):
        x = out.detach()
        if x.dim() == 3:  # [B,S,H] -> [T,H]
            x = x.reshape(-1, x.shape[-1])
        if self.to_cpu:
            x = x.cpu()
        self.cache[name].append(x)

    def add_hook(self, full):
        for name in self.names:
            self.handles.append(
                full[name].register_forward_hook(
                    functools.partial(self._hook, name=name)
                )
            )

    def clear_hook(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def clear_cache(self):
        for name in self.names:
            self.cache[name].clear()
