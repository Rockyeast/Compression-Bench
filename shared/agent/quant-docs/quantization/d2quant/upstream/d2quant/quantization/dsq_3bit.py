import torch


@torch.no_grad()
def fit_dual_scale_3bit(
    x: torch.Tensor,
    iter: int = 15,
    eps: float = 1e-8,
    do_normalize: bool = False,
    ridge: float = 0.0,  # optional small ridge for numerical stability
):
    """
    Fit the 3-bit Dual-Scale Quantizer with one shared column scale.

    Model:
        x_hat[i,j] = c[j] * ( r0[i]*B0[i,j] + r1[i]*B1[i,j] + r2[i]*B2[i,j] )
    where:
        B0,B1,B2 in {+1,-1}
        r0,r1,r2 > 0 (row scales)
        c > 0 (shared col scales)

    C-first update order per-iter:
        (1) joint solve r0,r1,r2 (3x3 normal eq) + fold sign into B
        (2) update shared c (per-column LS)
        (3) discrete update (B0,B1,B2) by 8-way nearest (search)

    Return:
        r0,r1,r2: [m]
        c:        [n]
        B0,B1,B2: [m,n] in {+1,-1}
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    X = x.detach().to(dtype=torch.float32)
    dev = X.device
    m, n = X.shape

    # ------------------------------------------------------------
    # helper: joint solve r0,r1,r2 per-row (3x3 normal equations)
    #         + fold sign into B so r's >= 0
    # ------------------------------------------------------------
    def update_r012_joint(X, c, B0, B1, B2, eps: float, ridge: float = 0.0):
        # ensure +/-1
        B0 = B0.clone()
        B1 = B1.clone()
        B2 = B2.clone()
        B0[B0 == 0] = 1.0
        B1[B1 == 0] = 1.0
        B2[B2 == 0] = 1.0

        c = c.to(dtype=torch.float32)
        c2 = c * c  # [n]

        # Diagonal terms are identical with +/-1: sum c^2
        A = c2.sum()  # scalar

        # Off-diagonal terms per row:
        S01 = (c2[None, :] * (B0 * B1)).sum(dim=1)  # [m]
        S02 = (c2[None, :] * (B0 * B2)).sum(dim=1)  # [m]
        S12 = (c2[None, :] * (B1 * B2)).sum(dim=1)  # [m]

        # RHS per row:
        p0 = (c[None, :] * B0 * X).sum(dim=1)  # [m]
        p1 = (c[None, :] * B1 * X).sum(dim=1)  # [m]
        p2 = (c[None, :] * B2 * X).sum(dim=1)  # [m]

        # Build batched 3x3 normal matrices G: [m,3,3]
        G = torch.empty((m, 3, 3), device=dev, dtype=torch.float32)
        diag = A + ridge
        G[:, 0, 0] = diag
        G[:, 1, 1] = diag
        G[:, 2, 2] = diag
        G[:, 0, 1] = S01
        G[:, 1, 0] = S01
        G[:, 0, 2] = S02
        G[:, 2, 0] = S02
        G[:, 1, 2] = S12
        G[:, 2, 1] = S12

        rhs = torch.stack([p0, p1, p2], dim=1)  # [m,3]

        # Solve G * r = rhs (batched)
        # torch.linalg.solve expects [m,3,3] and [m,3] -> [m,3]
        r_raw = torch.linalg.solve(G, rhs)

        r0_raw = r_raw[:, 0]
        r1_raw = r_raw[:, 1]
        r2_raw = r_raw[:, 2]

        # fold sign into B so r>=0
        s0 = torch.sign(r0_raw)
        s0[s0 == 0] = 1.0
        r0 = r0_raw.abs().clamp_min(eps)
        B0 = B0 * s0[:, None]

        s1 = torch.sign(r1_raw)
        s1[s1 == 0] = 1.0
        r1 = r1_raw.abs().clamp_min(eps)
        B1 = B1 * s1[:, None]

        s2 = torch.sign(r2_raw)
        s2[s2 == 0] = 1.0
        r2 = r2_raw.abs().clamp_min(eps)
        B2 = B2 * s2[:, None]

        return r0, r1, r2, B0, B1, B2

    # -------------------------
    # init (greedy residual, shared c)
    # -------------------------
    c = X.abs().mean(dim=0).clamp_min(eps)  # [n]

    # term 0
    B0 = torch.sign(X)
    B0 = torch.where(B0 == 0, torch.ones_like(B0), B0)
    T0 = c[None, :] * B0
    r0 = (
        ((X * T0).sum(dim=1) / (T0 * T0).sum(dim=1).clamp_min(eps)).abs().clamp_min(eps)
    )
    X0 = c[None, :] * (r0[:, None] * B0)
    R1 = X - X0

    # term 1
    B1 = torch.sign(R1)
    B1 = torch.where(B1 == 0, torch.ones_like(B1), B1)
    T1 = c[None, :] * B1
    r1 = (
        ((R1 * T1).sum(dim=1) / (T1 * T1).sum(dim=1).clamp_min(eps))
        .abs()
        .clamp_min(eps)
    )
    X1 = c[None, :] * (r1[:, None] * B1)
    R2 = R1 - X1

    # term 2
    B2 = torch.sign(R2)
    B2 = torch.where(B2 == 0, torch.ones_like(B2), B2)
    T2 = c[None, :] * B2
    r2 = (
        ((R2 * T2).sum(dim=1) / (T2 * T2).sum(dim=1).clamp_min(eps))
        .abs()
        .clamp_min(eps)
    )

    if do_normalize:
        a = c.mean().clamp_min(eps)
        c = c / a
        r0 = r0 * a
        r1 = r1 * a
        r2 = r2 * a

    # Precompute sign table for 8 candidates (idx 0..7)
    # Each row is (s0,s1,s2) in {-1,+1}
    sign_table = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        device=dev,
        dtype=torch.float32,
    )  # [8,3]

    # -------------------------
    # ALS + discrete update (C-first)
    # -------------------------
    for _ in range(int(iter)):
        # (1') joint update r0,r1,r2 (3x3) + fold sign into B's
        r0, r1, r2, B0, B1, B2 = update_r012_joint(
            X, c, B0, B1, B2, eps=eps, ridge=ridge
        )

        # (2) update shared c (per-column LS)
        S = r0[:, None] * B0 + r1[:, None] * B1 + r2[:, None] * B2  # [m,n]
        num = (X * S).sum(dim=0)
        den = (S * S).sum(dim=0).clamp_min(eps)
        c = (num / den).abs().clamp_min(eps)

        # (3) discrete update (B0,B1,B2) by 8-way nearest
        comb0 = r0[:, None] * c[None, :]  # [m,n]
        comb1 = r1[:, None] * c[None, :]  # [m,n]
        comb2 = r2[:, None] * c[None, :]  # [m,n]

        # 8 candidates: ±comb0 ±comb1 ±comb2
        v = torch.stack(
            [
                -comb0 - comb1 - comb2,
                -comb0 - comb1 + comb2,
                -comb0 + comb1 - comb2,
                -comb0 + comb1 + comb2,
                comb0 - comb1 - comb2,
                comb0 - comb1 + comb2,
                comb0 + comb1 - comb2,
                comb0 + comb1 + comb2,
            ],
            dim=-1,
        )  # [m,n,8]

        idx = torch.argmin((X.unsqueeze(-1) - v).abs(), dim=-1)  # [m,n] in [0..7]

        # map idx -> (B0,B1,B2) using sign_table
        # gather signs for each element
        s012 = sign_table[idx]  # [m,n,3]
        B0 = s012[..., 0]
        B1 = s012[..., 1]
        B2 = s012[..., 2]

        if do_normalize:
            a = c.mean().clamp_min(eps)
            c = c / a
            r0 = r0 * a
            r1 = r1 * a
            r2 = r2 * a

    return r0, r1, r2, c, B0, B1, B2


class DualScaleQuantizer(torch.nn.Module):
    """
    Three-bit Dual-Scale Quantizer with one shared column scale.

    After find_params(x2d), cached params:
        row_scale1, row_scale2, row_scale3: [out]
        col_scale:                          [in] (shared)
    Quantize outputs float reconstructed weights (same dtype as input).

    This is an 8-level (3-bit) structure: candidates are ±a0 ±a1 ±a2.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("row_scale1", torch.empty(0))
        self.register_buffer("row_scale2", torch.empty(0))
        self.register_buffer("row_scale3", torch.empty(0))
        self.register_buffer("col_scale", torch.empty(0))
        self.bits = 3
        self.num_iters = 15
        self.ridge = 0.0
        self.do_normalize = False
        self._ready = False

    def ready(self):
        return self._ready

    def configure(
        self,
        bits: int = 3,
        num_iters: int = 15,
        ridge: float = 0.0,
        do_normalize: bool = False,
    ):
        assert bits == 3, "DSQ order 3 requires 3-bit (8-level) quantization."
        self.bits = int(bits)
        self.num_iters = int(num_iters)
        self.ridge = float(ridge)
        self.do_normalize = bool(do_normalize)

    @torch.no_grad()
    def find_params(self, x2d: torch.Tensor):
        """
        x2d: [out, in] (or [out, block_cols])  2D weight block or full layer
        """
        assert x2d.dim() == 2, f"find_params expects 2D, got {x2d.dim()}D"
        dev = x2d.device

        r0, r1, r2, c, _, _, _ = fit_dual_scale_3bit(
            x2d.float(),
            iter=self.num_iters,
            eps=1e-8,
            do_normalize=self.do_normalize,
            ridge=self.ridge,
        )

        self.row_scale1 = r0.to(device=dev, dtype=torch.float32)
        self.row_scale2 = r1.to(device=dev, dtype=torch.float32)
        self.row_scale3 = r2.to(device=dev, dtype=torch.float32)
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
        if not (self.ready() and self.bits == 3):
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
            a2 = self.row_scale3.to(device=dev, dtype=torch.float32) * c  # [out]

            # candidates: ±a0 ±a1 ±a2 (8 values)
            v = torch.stack(
                [
                    -a0 - a1 - a2,
                    -a0 - a1 + a2,
                    -a0 + a1 - a2,
                    -a0 + a1 + a2,
                    a0 - a1 - a2,
                    a0 - a1 + a2,
                    a0 + a1 - a2,
                    a0 + a1 + a2,
                ],
                dim=1,
            )  # [out,8]

            idx = torch.argmin((x_fp_1d[:, None] - v).abs(), dim=1)  # [out]
            q_1d = v.gather(1, idx[:, None]).squeeze(1)  # [out]
            q = q_1d.reshape_as(x_fp)
            return q.to(dtype=x_dtype)

        # ---------- case 2: 2D block ----------
        assert x_fp.dim() == 2, f"col_idx is None => expect 2D block, got {x_fp.dim()}D"
        out_dim, ncol = x_fp.shape

        c = (
            self.col_scale[:ncol].to(device=dev, dtype=torch.float32).view(1, ncol)
        )  # [1,ncol]

        r0 = self.row_scale1.to(device=dev, dtype=torch.float32).view(-1, 1)  # [out,1]
        r1 = self.row_scale2.to(device=dev, dtype=torch.float32).view(-1, 1)  # [out,1]
        r2 = self.row_scale3.to(device=dev, dtype=torch.float32).view(-1, 1)  # [out,1]

        a0 = r0 * c  # [out,ncol]
        a1 = r1 * c  # [out,ncol]
        a2 = r2 * c  # [out,ncol]

        v = torch.stack(
            [
                -a0 - a1 - a2,
                -a0 - a1 + a2,
                -a0 + a1 - a2,
                -a0 + a1 + a2,
                a0 - a1 - a2,
                a0 - a1 + a2,
                a0 + a1 - a2,
                a0 + a1 + a2,
            ],
            dim=-1,
        )  # [out,ncol,8]

        idx = torch.argmin((x_fp.unsqueeze(-1) - v).abs(), dim=-1)  # [out,ncol]
        q = v.gather(-1, idx.unsqueeze(-1)).squeeze(-1)  # [out,ncol]

        return q.to(dtype=x_dtype)
