import torch


@torch.no_grad()
def log_snr_from_lists(
    y_student_list,
    y_teacher_list,
    logger,
    tag: str = "",
    eps: float = 1e-8,
):
    """
    Compute per-dim SNR for diff = y_teacher - y_student:
      mu = E[diff], std = sqrt(Var(diff)), snr = |mu|/(std+eps)

    y_*_list: list of [T,H] tensors (or [B,S,H]/[S,H]/[H])
    """
    if logger is None:
        return
    if (
        y_student_list is None
        or y_teacher_list is None
        or len(y_student_list) == 0
        or len(y_teacher_list) == 0
    ):
        logger.debug(f"[SNR][{tag}] empty lists, skip.")
        return

    assert len(y_student_list) == len(y_teacher_list), (
        len(y_student_list),
        len(y_teacher_list),
    )

    # infer device from first teacher tensor
    dev = y_teacher_list[0].device
    sum1 = None
    sum2 = None
    n_total = 0

    def to_2d(x):
        x = x.detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() == 2:
            pass
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        else:
            raise RuntimeError(f"[SNR][{tag}] unexpected dim={x.dim()}")
        return x

    for ys, yt in zip(y_student_list, y_teacher_list):
        ys = to_2d(ys).to(dev, dtype=torch.float32)
        yt = to_2d(yt).to(dev, dtype=torch.float32)
        assert ys.shape == yt.shape, (
            f"[SNR][{tag}] shape mismatch: {ys.shape} vs {yt.shape}"
        )

        diff = yt - ys  # [T,H]
        if sum1 is None:
            H = diff.shape[1]
            sum1 = torch.zeros(H, device=dev, dtype=torch.float64)
            sum2 = torch.zeros(H, device=dev, dtype=torch.float64)

        sum1 += diff.sum(dim=0).to(torch.float64)
        sum2 += (diff * diff).sum(dim=0).to(torch.float64)
        n_total += diff.shape[0]

    if n_total == 0:
        logger.debug(f"[SNR][{tag}] n_total=0, skip.")
        return

    mu = (sum1 / n_total).to(torch.float32)
    ex2 = (sum2 / n_total).to(torch.float32)
    var = (ex2 - mu * mu).clamp_min(0.0)
    std = torch.sqrt(var + eps)
    snr = mu.abs() / (std + eps)

    # basic stats
    snr_min = snr.min().item()
    snr_max = snr.max().item()
    snr_mean = snr.mean().item()
    snr_std = snr.std(unbiased=False).item()

    # robust quantiles
    q = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=dev)
    qs = torch.quantile(snr, q).tolist()

    # optional: theoretical max removable MSE ratio by constant bias (for intuition)
    snr_max2 = snr_max * snr_max
    explain_max = snr_max2 / (1.0 + snr_max2)

    logger.debug(
        f"[SNR][{tag}] stats: min={snr_min:.4f}, max={snr_max:.4f}, "
        f"mean={snr_mean:.4f}, std={snr_std:.4f}, max_explain≈{explain_max * 100:.2f}%"
    )
    logger.debug(
        f"[SNR][{tag}] quantiles: "
        f"p0={qs[0]:.4f}, p1={qs[1]:.4f}, p10={qs[2]:.4f}, p50={qs[3]:.4f}, "
        f"p90={qs[4]:.4f}, p99={qs[5]:.4f}, p100={qs[6]:.4f}"
    )
