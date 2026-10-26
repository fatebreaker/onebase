"""Error-compensated merge onto a frozen grid (GPTQ, Frantar et al. 2023, with the checkpoint's own scales fixed).

A nearest merge rounds every entry of W_q + dW onto the base's grid on its own, so an update below half a step is
erased. Here the columns are rounded one at a time and each column's rounding error is pushed onto the columns not yet
rounded through the inverse Hessian H = X^T X of the layer's inputs (GPTQ's update), so that what is kept small is the
error in the layer's outputs X(Q - (W_q + dW))^T, not in each weight. The grid never changes: the block scales of the
checkpoint (NVFP4) or its block absmax (NF4, as QLoRA stores the base) stay as they are, and every merged entry is a code
of that grid, so the result is a valid checkpoint of the same format and size.

    base_q, merged = gptq_frozen(W, dW, H, fmt="nvfp4")      # W [out, in] (bf16 master), dW the update, H [in, in]
"""
from __future__ import annotations

import torch

from . import quant


def frozen_grid(W: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """The base on its grid and the per-entry grid of the checkpoint: (base_q [out, in], per-block scale
    [out, n_blocks], block size, levels in units of the scale)."""
    W = W.float()
    if fmt == "nvfp4":
        base_q, scale, pad = quant.fp4_quant_with_scale(W, "nvfp4")
        if pad:
            raise ValueError("input width must be a multiple of 16")
        levels = torch.cat([-quant._E2M1_POS.flip(0)[:-1], quant._E2M1_POS])
        return base_q, scale.squeeze(-1), 16, levels
    if fmt == "nf4":
        g, pad = quant._grouped(W, 64)
        if pad:
            raise ValueError("input width must be a multiple of 64")
        amax = g.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax, torch.ones_like(amax))
        base_q = quant.nf4_quant(W)
        return base_q, scale.squeeze(-1), 64, quant.NF4_LEVELS.clone()
    if fmt == "fp8":     # per-row E4M3 (quant.fp8_quant): one block per row, its row scale kept
        base_q, scale = quant.fp8_quant(W)
        return base_q, scale.reshape(W.shape[0], 1), W.shape[1], quant._e4m3_levels().clone()
    raise ValueError(f"gptq_frozen supports nvfp4, nf4 and fp8, not {fmt!r}")


def _round_col(w: torch.Tensor, s: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Round one column onto its rows' fixed grids (nearest level of levels * s; values beyond the end saturate)."""
    return quant._round_to_levels(w / s, levels, "nearest", None) * s


@torch.no_grad()
def gptq_frozen(W: torch.Tensor, dW: torch.Tensor, H: torch.Tensor, fmt: str = "nvfp4", damp: float = 0.01,
                blocksize: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ onto the frozen grid of the checkpoint quantized from W: returns (base_q, merged), both float32 [out, in].
    The target is base_q + dW, the stored checkpoint plus the update (what a nearest frozen merge rounds)."""
    base_q, scale, bs, levels = frozen_grid(W, fmt)
    return base_q, gptq_on_grid(base_q + dW.float(), H, scale, bs, levels, damp, blocksize)


@torch.no_grad()
def gptq_on_grid(T: torch.Tensor, H: torch.Tensor, scale: torch.Tensor, bs: int, levels: torch.Tensor,
                 damp: float = 0.01, blocksize: int = 128) -> torch.Tensor:
    """GPTQ of the target T [out, in] onto a given grid: entry (i, j) may take levels * scale[i, j // bs] (e.g. an
    official NVFP4 release's E2M1 codes times its block scale times its tensor scale). Returns float32 [out, in]."""
    levels = levels.to(T.device, torch.float32)
    T = T.float().clone()
    n = T.shape[1]
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(T)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        T1 = T[:, i1:i2].clone()
        Err1 = torch.zeros_like(T1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            q = _round_col(T1[:, i], scale[:, j // bs], levels)
            Q[:, j] = q
            err = (T1[:, i] - q) / Hinv1[i, i]
            T1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        T[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q
