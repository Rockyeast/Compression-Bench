from d2quant import config as utils
from d2quant import model_utils

import torch
import typing
import tqdm
from .hadamard import random_hadamard_matrix, random_walsh_matrix, dct_matrix


def _right_rotate_groupwise(
    W: torch.Tensor, Q: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    Apply a groupwise right rotation: W[:, group] @ Q.

    W: [out_dim, dim]
    Q: [group_size, group_size], shared by all groups.
    """
    out_dim, dim = W.shape
    G = group_size
    assert Q.shape == (G, G), f"Q shape {Q.shape} must be (group_size, group_size)"
    assert dim % G == 0, f"dim {dim} not divisible by group_size {G}"
    num_groups = dim // G

    W_dev = W.to(device=utils.DEV, dtype=torch.float64)
    Q_dev = Q.to(device=utils.DEV, dtype=torch.float64)
    out = torch.empty_like(W_dev)

    for g in range(num_groups):
        cs = g * G
        ce = (g + 1) * G
        W_block = W_dev[:, cs:ce]  # [out_dim, G]
        out[:, cs:ce] = W_block @ Q_dev  # [out_dim, G]

    return out


def _left_rotate_groupwise(
    W: torch.Tensor, Q: torch.Tensor, group_size: int
) -> torch.Tensor:
    """
    Apply a groupwise left rotation: Q.T @ W[row_group, :].

    W: [dim, in_dim]
    Q: [group_size, group_size], shared by all groups.
    """
    dim, in_dim = W.shape
    G = group_size
    assert Q.shape == (G, G), f"Q shape {Q.shape} must be (group_size, group_size)"
    assert dim % G == 0, f"dim {dim} not divisible by group_size {G}"
    num_groups = dim // G

    W_dev = W.to(device=utils.DEV, dtype=torch.float64)
    Q_dev = Q.to(device=utils.DEV, dtype=torch.float64)
    out = torch.empty_like(W_dev)

    for g in range(num_groups):
        rs = g * G
        re = (g + 1) * G
        W_block = W_dev[rs:re, :]  # [G, in_dim]
        out[rs:re, :] = Q_dev.t() @ W_block  # [G, in_dim]

    return out


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    with torch.no_grad():
        if hasattr(layernorm, "weight") and layernorm.weight is not None:
            layernorm.weight.data.fill_(1.0)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            layernorm.bias.data.zero_()


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model, args):

    model_type = model_utils.get_model_type(model)

    kwargs = {"model": model, "model_type": model_type}

    # Embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_utils.get_transformer_layers(**kwargs)

    # Fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:
        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(
                layer.self_attn_layer_norm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        elif (
            model_type == model_utils.QWEN3_MODEL
            or model_type == model_utils.QWEN2_MODEL
        ):
            fuse_ln_linear(
                layer.input_layernorm,
                [
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                ],
            )
            fuse_ln_linear(
                layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj]
            )
        else:
            raise ValueError(f"Unknown model type {model_type}")
        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(
        model_utils.get_pre_head_layernorm(**kwargs),
        [model_utils.get_lm_head(**kwargs)],
    )

    # if args.update_ln_mean or args.update_ln:
    #     model_utils.replace_modules(
    #         model,
    #         transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
    #         lambda _: model_utils.RMSNWithBias(model.config.hidden_size),
    #         replace_layers=False,
    #     )


def random_orthogonal_matrix(size, device, seed=42):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    elif mode == "walsh":
        return random_walsh_matrix(size, device)
    elif mode == "dct":
        return dct_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type):
    # Rotate the MLP input weights.
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f"Unknown model type {model_type}")
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, Q, model_type):
    # Rotate the MLP output weights and bias.
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(
        model, model_type=model_utils.model_type_extractor(model)
    )
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_embeddings_groupwise(model, Q: torch.Tensor, group_size: int) -> None:
    """Groupwise right rotation for embeddings: W_emb = W_emb @ Q_block_diag"""
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data
        W_rot = _right_rotate_groupwise(W_, Q, group_size)
        W.weight.data = W_rot.to(device="cpu", dtype=dtype)


def rotate_attention_inputs_groupwise(layer, Q, model_type, group_size: int) -> None:
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data
        W_rot = _right_rotate_groupwise(W_, Q, group_size)
        W.weight.data = W_rot.to(device="cpu", dtype=dtype)


def rotate_attention_output_groupwise(layer, Q, model_type, group_size: int) -> None:
    """
    W_o = Q_block_diag^T @ W_o
    """
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        W = layer.self_attn.o_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    dtype = W.weight.data.dtype
    W_ = W.weight.data
    W_rot = _left_rotate_groupwise(W_, Q, group_size)
    W.weight.data = W_rot.to(device="cpu", dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)  # [dim]
        dim = b.numel()
        assert Q.shape == (dim, dim)
        assert dim % group_size == 0
        num_groups = dim // group_size
        b_out = torch.empty_like(b)
        for g in range(num_groups):
            s = g * group_size
            e = (g + 1) * group_size
            Q_block = Q[s:e, s:e].to(utils.DEV, torch.float64)
            b_out[s:e] = Q_block.t() @ b[s:e]
        W.bias.data = b_out.to(device="cpu", dtype=dtype)


def rotate_mlp_input_groupwise(layer, Q, model_type, group_size: int):
    """
    W_in = W_in @ Q_block_diag
    """
    if model_type == model_utils.LLAMA_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    elif model_type == model_utils.OPT_MODEL:
        mlp_inputs = [layer.fc1]
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    else:
        raise ValueError(f"Unknown model type {model_type}")

    for W in mlp_inputs:
        dtype = W.weight.data.dtype
        W_ = W.weight.data
        W_rot = _right_rotate_groupwise(W_, Q, group_size)
        W.weight.data = W_rot.to(device="cpu", dtype=dtype)


def rotate_mlp_output_groupwise(layer, Q, model_type, group_size: int):
    """
    W_out = Q_block_diag^T @ W_out
    """
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.fc2
    elif model_type == model_utils.QWEN3_MODEL or model_type == model_utils.QWEN2_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f"Unknown model type {model_type}")

    dtype = W.weight.data.dtype
    W_ = W.weight.data
    W_rot = _left_rotate_groupwise(W_, Q, group_size)
    W.weight.data = W_rot.to(device="cpu", dtype=dtype)

    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        dim = b.numel()
        assert Q.shape == (dim, dim)
        assert dim % group_size == 0
        num_groups = dim // group_size
        b_out = torch.empty_like(b)
        for g in range(num_groups):
            s = g * group_size
            e = (g + 1) * group_size
            Q_block = Q[s:e, s:e].to(utils.DEV, torch.float64)
            b_out[s:e] = Q_block.t() @ b[s:e]
        W.bias.data = b_out.to(device="cpu", dtype=dtype)


def rotate_head_groupwise(model, Q: torch.Tensor, group_size: int) -> None:
    """
    W_head = W_head @ Q_block_diag
    """
    W = model_utils.get_lm_head(
        model, model_type=model_utils.model_type_extractor(model)
    )
    dtype = W.weight.data.dtype
    W_ = W.weight.data
    W_rot = _right_rotate_groupwise(W_, Q, group_size)
    W.weight.data = W_rot.to(device="cpu", dtype=dtype)


@torch.inference_mode()
def rotate_model(model, args):

    if args.groupwise_rot:
        Q_group = get_orthogonal_matrix(args.w_groupsize, args.rotate_mode)
    else:
        Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)

    model_type = model_utils.model_type_extractor(model)

    with torch.inference_mode():
        if args.groupwise_rot:
            rotate_embeddings_groupwise(model, Q_group, args.w_groupsize)
            rotate_head_groupwise(model, Q_group, args.w_groupsize)
            utils.cleanup_memory()
        else:
            rotate_embeddings(model, Q)
            rotate_head(model, Q)
            utils.cleanup_memory()

    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.groupwise_rot:
            rotate_attention_inputs_groupwise(
                layers[idx], Q_group, model_type, args.w_groupsize
            )
            rotate_attention_output_groupwise(
                layers[idx], Q_group, model_type, args.w_groupsize
            )
            rotate_mlp_input_groupwise(
                layers[idx], Q_group, model_type, args.w_groupsize
            )
            rotate_mlp_output_groupwise(
                layers[idx], Q_group, model_type, args.w_groupsize
            )
        else:
            rotate_attention_inputs(layers[idx], Q, model_type)
            rotate_attention_output(layers[idx], Q, model_type)
            rotate_mlp_input(layers[idx], Q, model_type)
            rotate_mlp_output(layers[idx], Q, model_type)
