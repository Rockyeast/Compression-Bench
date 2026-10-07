import torch
import torch.nn as nn
import types


def _ensure_rmsnorm_bias(rms: nn.Module):
    """
    In-place: make `rms` always output orig_rms(x) + bias.
    This is permanent for this module object (until you manually revert).
    """
    assert hasattr(rms, "weight") and rms.weight is not None, "rms must have weight"
    H = rms.weight.numel()
    device = rms.weight.device
    dtype = rms.weight.dtype

    # 1) add bias param if not exist
    if not hasattr(rms, "bias") or rms.bias is None:
        init = torch.zeros(H, device=device, dtype=dtype)
        rms.bias = nn.Parameter(init)  # registered => appears in state_dict

    # 2) patch forward only once
    if not hasattr(rms, "_orig_forward"):
        rms._orig_forward = rms.forward  # stash original

        def _forward_with_bias(self, x: torch.Tensor):
            y = self._orig_forward(x)
            b = self.bias.view(*([1] * (y.dim() - 1)), -1)  # broadcast over [B,S,...]
            return y + b

        rms.forward = types.MethodType(_forward_with_bias, rms)

    return rms


def apply_deviation_aware_correction(
    rms: torch.nn.Module,
    y_student_list,  # each [T,H]
    y_teacher_list,  # each [T,H]
    clip: float = 10.0,
    lmbd: float = 1.0,
    # ---- selection options (hard) ----
    snr_th: float = None,  # keep channels with snr >= snr_th
    keep_ratio: float = None,  # keep top-k by snr (k = ceil(H*keep_ratio))
    explain_th: float = None,  # NEW: keep channels with explain >= explain_th
    eps: float = 1e-8,
    return_extra: bool = False,  # if True -> return (bias, snr, mask, th_used, explain)
):
    """
    Align output mean (bias) by minimizing || y_s + b - y_t ||^2.

    diff = y_teacher - y_student, per-channel:
        mu  = E[diff]
        std = sqrt(Var(diff))
        snr = |mu| / (std + eps)

    NEW: explain ratio (how much MSE can be reduced by optimal constant bias):
        explain = snr^2 / (1 + snr^2)   in [0,1]
    If explain_th is provided: keep channels with explain >= explain_th.

    Priority (if multiple provided):
        explain_th > snr_th > keep_ratio > keep_all
    """
    _ensure_rmsnorm_bias(rms)
    assert hasattr(rms, "bias") and rms.bias is not None, "RMSNorm must have bias"
    assert len(y_student_list) == len(y_teacher_list), (
        len(y_student_list),
        len(y_teacher_list),
    )
    assert len(y_student_list) > 0

    device = rms.weight.device
    H = rms.weight.numel()

    # accumulate over ALL tokens in ALL samples
    sum_diff = torch.zeros(H, device=device, dtype=torch.float32)
    sumsq_diff = torch.zeros(H, device=device, dtype=torch.float32)
    cnt = 0

    for ys, yt in zip(y_student_list, y_teacher_list):
        assert ys.shape == yt.shape, f"Shape mismatch: {ys.shape} vs {yt.shape}"
        ys = ys.to(device=device, dtype=torch.float32)
        yt = yt.to(device=device, dtype=torch.float32)

        diff = yt - ys  # [T,H]
        sum_diff += diff.sum(dim=0)  # [H]
        sumsq_diff += (diff * diff).sum(dim=0)
        cnt += diff.shape[0]

    cnt = max(int(cnt), 1)
    mean = sum_diff / cnt  # [H]
    var = (sumsq_diff / cnt) - (mean * mean)  # [H]
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var) + float(eps)  # [H]
    snr = mean.abs() / std  # [H]

    # NEW: explain ratio in [0,1]
    snr2 = snr * snr
    explain = snr2 / (1.0 + snr2)

    # ---- hard selection mask ----
    th_used = None
    mode_used = "all"

    if explain_th is not None:
        th_used = float(explain_th)
        # # allow passing 0~100 as percentage too
        # if th_used > 1.0:
        #     th_used = th_used / 100.0
        th_used = min(max(th_used, 0.0), 1.0)
        mask = explain >= th_used
        mode_used = "explain_th"

    elif snr_th is not None:
        th_used = float(snr_th)
        mask = snr >= th_used
        mode_used = "snr_th"

    elif keep_ratio is not None:
        keep_ratio = float(keep_ratio)
        keep_ratio = min(max(keep_ratio, 0.0), 1.0)
        k = int(torch.ceil(torch.tensor(H * keep_ratio, device=device)).item())
        k = max(k, 1) if keep_ratio > 0 else 0
        if k <= 0:
            mask = torch.zeros(H, device=device, dtype=torch.bool)
            th_used = float("inf")
        else:
            topk_vals = torch.topk(snr, k, largest=True).values
            th_used = topk_vals.min().item()
            mask = snr >= th_used
        mode_used = "keep_ratio"

    else:
        mask = torch.ones(H, device=device, dtype=torch.bool)
        th_used = None
        mode_used = "all"

    # apply mask (hard gating)
    bias = mean * mask.to(mean.dtype)

    # scale + clip
    bias = bias * float(lmbd)
    bias = bias.clamp(min=-clip, max=clip)

    # update
    rms.bias.add_(bias.to(rms.bias.dtype))

    if return_extra:
        # th_used: explain_th / snr_th / snr-threshold-from-keep_ratio / None
        return bias, snr, mask, th_used, explain, mode_used
    return bias


@torch.no_grad()
def apply_mean_shift(
    rms: torch.nn.Module,
    y_student_list,  # each [T,H]
    y_teacher_list,  # each [T,H], already processed
    clip: float = 10.0,  # clip bias values
    lmbd: float = 1.0,
):
    """
    Align the output mean (bias) by minimizing || y_s + b - y_t ||^2
    - Updates the bias parameter in RMSNorm with the computed bias.
    - bias = mean(y_t - y_s)
    """
    _ensure_rmsnorm_bias(rms)
    assert hasattr(rms, "bias") and rms.bias is not None, "RMSNorm must have bias"
    assert len(y_student_list) == len(y_teacher_list), (
        len(y_student_list),
        len(y_teacher_list),
    )
    assert len(y_student_list) > 0

    device = rms.weight.device
    H = rms.weight.numel()

    b_sum = torch.zeros(H, device=device, dtype=torch.float32)
    n_cnt = 0

    # Iterate over each sample in the lists
    for ys, yt in zip(y_student_list, y_teacher_list):
        # Ensure the shapes are correct
        assert ys.shape == yt.shape, f"Shape mismatch: {ys.shape} vs {yt.shape}"

        ys = ys.to(device=device, dtype=torch.float32)
        yt = yt.to(device=device, dtype=torch.float32)

        # Compute the per-channel mean difference (bias)
        b_sum += (yt - ys).mean(dim=0)  # [H]
        n_cnt += 1

    # Compute the final bias
    bias = b_sum / max(n_cnt, 1)
    bias = bias.clamp(min=-clip, max=clip)

    bias = (bias * float(lmbd)).clamp(min=-clip, max=clip)

    # Update RMSNorm bias parameter with the computed bias
    with torch.no_grad():
        rms.bias.add_(bias.to(rms.bias.dtype))
        # rms.bias.data.copy_(bias.to(rms.bias.dtype))

    return bias
