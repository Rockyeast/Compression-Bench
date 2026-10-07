import torch


def compute_local_perms(diag_h, group_size):
    """Sort columns within each quantization group by Hessian importance."""
    num_groups = diag_h.numel() // group_size
    return [
        torch.argsort(
            diag_h[group * group_size : (group + 1) * group_size],
            descending=True,
        )
        for group in range(num_groups)
    ]


def compute_global_perm(diag_h, group_size):
    """Sort quantization groups by their maximum Hessian importance."""
    num_groups = diag_h.numel() // group_size
    metrics = torch.tensor(
        [
            diag_h[group * group_size : (group + 1) * group_size].max().item()
            for group in range(num_groups)
        ],
        device=diag_h.device,
    )
    return torch.argsort(metrics, descending=True)


def compose_final_perm(local_perms, global_perm, group_size):
    """Compose local and global group permutations."""
    final_perm = []
    for original_group in global_perm.tolist():
        offset = original_group * group_size
        final_perm.extend((local_perms[original_group] + offset).tolist())
    return torch.tensor(final_perm, dtype=torch.long, device=global_perm.device)


def invert_perm(perm):
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.numel(), device=perm.device)
    return inverse
