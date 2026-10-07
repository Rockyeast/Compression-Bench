import torch


@torch.no_grad()
def fit_dual_scale_2bit(
    x: torch.Tensor,
    iter: int = 15,
    eps: float = 1e-8,
    do_normalize: bool = False,
    ridge: float = 0.0,  # optional small ridge for numerical stability
):
    """
    Fit the 2-bit Dual-Scale Quantizer with one shared column scale.

    Model:
        x_hat[i,j] = c[j] * ( r0[i]*B0[i,j] + r1[i]*B1[i,j] )
    where:
        B0,B1 in {+1,-1}
        r0,r1 > 0 (row scales)
        c > 0 (shared col scales)

    Return:
        r0,r1: [m]
        c:     [n]
        B0,B1: [m,n] in {+1,-1}
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    X = x.detach().to(dtype=torch.float32)
    m, n = X.shape

    # -------------------------
    # helper: joint solve r0,r1 per-row (2x2 normal equations) + fold sign into B
    # -------------------------
    def update_r0_r1_joint(X, c, B0, B1, eps: float, ridge: float = 0.0):
        """
        For each row i solve:
            min_{r0,r1} || x_i - r0*(c*B0_i) - r1*(c*B1_i) ||_2^2
        Closed form from normal equations.

        Returns updated (r0, r1, B0, B1) with sign folded into B's so r's >= 0.
        """
        # ensure +/-1
        B0 = B0.clone()
        B1 = B1.clone()
        B0[B0 == 0] = 1.0
        B1[B1 == 0] = 1.0

        c = c.to(dtype=torch.float32)
        c2 = c * c  # [n]

        # With +/-1, per-row:
        # A = sum (c^2) ; D = sum (c^2) ; B_i = sum c^2 * (B0*B1)
        A = c2.sum() + ridge
        D = c2.sum() + ridge
        Bb = (c2[None, :] * (B0 * B1)).sum(dim=1)  # [m]

        # p_i = sum (c * B0 * X), q_i = sum (c * B1 * X)
        p = (c[None, :] * B0 * X).sum(dim=1)  # [m]
        q = (c[None, :] * B1 * X).sum(dim=1)  # [m]

        Delta = (A * D - Bb * Bb).clamp_min(eps)

        r0_raw = (D * p - Bb * q) / Delta
        r1_raw = (A * q - Bb * p) / Delta

        # fold sign into B so r>=0
        s0 = torch.sign(r0_raw)
        s0[s0 == 0] = 1.0
        r0 = r0_raw.abs().clamp_min(eps)
        B0 = B0 * s0[:, None]

        s1 = torch.sign(r1_raw)
        s1[s1 == 0] = 1.0
        r1 = r1_raw.abs().clamp_min(eps)
        B1 = B1 * s1[:, None]

        return r0, r1, B0, B1

    # -------------------------
    # -------------------------
    c = X.abs().mean(dim=0).clamp_min(eps)  # [n]

    B0 = torch.sign(X)
    B0 = torch.where(B0 == 0, torch.ones_like(B0), B0)

    # given (c, B0), fit r0 per-row (LS)
    T0 = c[None, :] * B0  # [m,n]
    num = (X * T0).sum(dim=1)
    den = (T0 * T0).sum(dim=1).clamp_min(eps)
    r0 = (num / den).abs().clamp_min(eps)  # [m]

    # init second term by residual
    X0 = c[None, :] * (r0[:, None] * B0)
    R1 = X - X0

    B1 = torch.sign(R1)
    B1 = torch.where(B1 == 0, torch.ones_like(B1), B1)

    T1 = c[None, :] * B1
    num = (R1 * T1).sum(dim=1)
    den = (T1 * T1).sum(dim=1).clamp_min(eps)
    r1 = (num / den).abs().clamp_min(eps)

    if do_normalize:
        a = c.mean().clamp_min(eps)
        c = c / a
        r0 = r0 * a
        r1 = r1 * a

    # -------------------------
    # -------------------------
    for _ in range(int(iter)):
        # (1') joint update r0,r1 (instead of two sequential LS)
        r0, r1, B0, B1 = update_r0_r1_joint(X, c, B0, B1, eps=eps, ridge=ridge)

        # (4) update shared c (per-column LS)
        # X ≈ c[j] * S[:,j],  S = r0*B0 + r1*B1
        S = r0[:, None] * B0 + r1[:, None] * B1  # [m,n]
        num = (X * S).sum(dim=0)
        den = (S * S).sum(dim=0).clamp_min(eps)
        c = (num / den).abs().clamp_min(eps)

        # (3) discrete update (B0,B1) by 4-way nearest
        # candidates: c*(±r0 ± r1)
        comb0 = r0[:, None] * c[None, :]  # [m,n]
        comb1 = r1[:, None] * c[None, :]  # [m,n]
        v = torch.stack(
            [-comb0 - comb1, -comb0 + comb1, comb0 - comb1, comb0 + comb1], dim=-1
        )  # [m,n,4]
        idx = torch.argmin((X.unsqueeze(-1) - v).abs(), dim=-1)  # [m,n]

        B0 = torch.ones_like(X)
        B0[(idx == 0) | (idx == 1)] = -1.0
        B1 = torch.ones_like(X)
        B1[(idx == 0) | (idx == 2)] = -1.0

        if do_normalize:
            a = c.mean().clamp_min(eps)
            c = c / a
            r0 = r0 * a
            r1 = r1 * a

    return r0, r1, c, B0, B1


class DualScaleQuantizer(torch.nn.Module):
    """
    Two-bit Dual-Scale Quantizer with one shared column scale.

    After find_params(x2d), cached params:
        row_scale1, row_scale2: [out]
        col_scale:             [in]  (shared for both B0/B1)
    Quantize outputs float reconstructed weights (same dtype as input).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("row_scale1", torch.empty(0))
        self.register_buffer("row_scale2", torch.empty(0))
        self.register_buffer("col_scale", torch.empty(0))
        self.bits = 2
        self.num_iters = 15
        self._ready = False

    def ready(self):
        return self._ready

    def configure(self, bits: int = 2, num_iters: int = 15):
        assert bits == 2, "DSQ order 2 requires 2-bit (4-level) quantization."
        self.bits = int(bits)
        self.num_iters = int(num_iters)

    @torch.no_grad()
    def find_params(self, x2d: torch.Tensor):
        """
        x2d: [out, in] (or [out, block_cols])  2D weight block or full layer
        """
        assert x2d.dim() == 2, f"find_params expects 2D, got {x2d.dim()}D"
        dev = x2d.device

        r0, r1, c, _, _ = fit_dual_scale_2bit(x2d.float(), iter=self.num_iters)

        self.row_scale1 = r0.to(device=dev, dtype=torch.float32)
        self.row_scale2 = r1.to(device=dev, dtype=torch.float32)
        self.col_scale = c.to(device=dev, dtype=torch.float32)
        self._ready = True

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, col_idx: int = None):
        """
        If col_idx is not None:
            x is a column, shape [out] or [out,1]
            use shared col_scale at col_idx, return reconstructed column
        Else:
            x is a 2D block, shape [out, ncol]
            use shared col_scale for all columns in the block, return reconstructed block

        NOTE (block case):
        """
        if not (self.ready() and self.bits == 2):
            return x

        x_dtype = x.dtype
        x_fp = x.detach().float()
        dev = x_fp.device

        # ---------- case 1: single column ----------
        if col_idx is not None:
            if x_fp.dim() == 2 and x_fp.shape[1] == 1:
                x_fp_1d = x_fp[:, 0]
            else:
                x_fp_1d = x_fp.reshape(-1)

            c = self.col_scale[col_idx].to(device=dev, dtype=torch.float32)  # scalar

            a0 = self.row_scale1.to(device=dev, dtype=torch.float32) * c  # [out]
            a1 = self.row_scale2.to(device=dev, dtype=torch.float32) * c  # [out]

            # candidates: c*(±r0 ± r1) = ±a0 ± a1
            v0 = -a0 - a1
            v1 = -a0 + a1
            v2 = a0 - a1
            v3 = a0 + a1

            V = torch.stack([v0, v1, v2, v3], dim=1)  # [out,4]
            idx = torch.argmin((x_fp_1d[:, None] - V).abs(), dim=1)  # [out]
            q_1d = V.gather(1, idx[:, None]).squeeze(1)  # [out]

            q = q_1d.reshape_as(x_fp)
            return q.to(dtype=x_dtype)

        # ---------- case 2: 2D block ----------
        assert x_fp.dim() == 2, f"col_idx is None => expect 2D block, got {x_fp.dim()}D"
        out_dim, ncol = x_fp.shape

        # shared col scales for this block: [1,ncol]
        c = (
            self.col_scale[:ncol].to(device=dev, dtype=torch.float32).view(1, ncol)
        )  # [1,ncol]

        r0 = self.row_scale1.to(device=dev, dtype=torch.float32).view(-1, 1)  # [out,1]
        r1 = self.row_scale2.to(device=dev, dtype=torch.float32).view(-1, 1)  # [out,1]

        a0 = r0 * c  # [out,ncol]
        a1 = r1 * c  # [out,ncol]

        v0 = -a0 - a1
        v1 = -a0 + a1
        v2 = a0 - a1
        v3 = a0 + a1

        V = torch.stack([v0, v1, v2, v3], dim=-1)  # [out,ncol,4]
        idx = torch.argmin((x_fp.unsqueeze(-1) - V).abs(), dim=-1)  # [out,ncol]
        q = V.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # [out,ncol]

        return q.to(dtype=x_dtype)
