"""GPTQ onto a frozen grid with act-order (lowbit-lora GPTQ-ACT): gptq_merge.gptq_on_grid, but the columns are rounded
in order of decreasing input energy (the diagonal of H), GPTQ's "act-order" (Frantar et al. 2023), so that the columns
whose rounding matters most are rounded while the most columns remain to absorb their error. Every column keeps the
scale of its own block in the original order, so the result lies on the same frozen grid as gptq_merge's.
Kept apart from gptq_merge.py, which running streams import.

    base_q, merged = gptq_frozen_act(W, dW, H, fmt="nvfp4")
    gain, err = output_stats(merged, base_q, dW, H)     # on the calibration inputs
"""
from __future__ import annotations

import torch

from . import gptq_merge as G


@torch.no_grad()
def _gptq_cols(T: torch.Tensor, H: torch.Tensor, S: torch.Tensor, levels: torch.Tensor, damp: float,
               blocksize: int) -> torch.Tensor:
    """gptq_merge.gptq_on_grid's loop with a scale per entry: column j may take levels * S[:, j]."""
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
            q = G._round_col(T1[:, i], S[:, j], levels)
            Q[:, j] = q
            err = (T1[:, i] - q) / Hinv1[i, i]
            T1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        T[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


@torch.no_grad()
def gptq_on_grid_act(T: torch.Tensor, H: torch.Tensor, scale: torch.Tensor, bs: int, levels: torch.Tensor,
                     damp: float = 0.01, blocksize: int = 128) -> torch.Tensor:
    """As gptq_merge.gptq_on_grid (entry (i, j) may take levels * scale[i, j // bs]), columns in act-order."""
    n = T.shape[1]
    perm = torch.argsort(torch.diag(H), descending=True)
    inv = torch.argsort(perm)
    S = scale.float()[:, torch.arange(n, device=scale.device) // bs][:, perm]
    Qp = _gptq_cols(T[:, perm], H[perm][:, perm], S, levels, damp, blocksize)
    return Qp[:, inv]


@torch.no_grad()
def gptq_frozen_act(W: torch.Tensor, dW: torch.Tensor, H: torch.Tensor, fmt: str = "nvfp4", damp: float = 0.01,
                    blocksize: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """gptq_merge.gptq_frozen with act-order: (base_q, merged), both float32 [out, in]."""
    base_q, scale, bs, levels = G.frozen_grid(W, fmt)
    return base_q, gptq_on_grid_act(base_q + dW.float(), H, scale, bs, levels, damp, blocksize)


@torch.no_grad()
def output_stats(Q: torch.Tensor, base_q: torch.Tensor, dW: torch.Tensor, H: torch.Tensor) -> tuple[float, float]:
    """On the calibration inputs X (H = X^T X): the merge's output gain <X D^T, X dW^T> / ||X dW^T||^2, D = Q - base_q,
    and its output error ||X (D - dW)^T|| / ||X dW^T||."""
    D = (Q - base_q).double()
    d = dW.double()
    Hd = H.double()
    den = ((d @ Hd) * d).sum()
    E = D - d
    return float(((D @ Hd) * d).sum() / den), float((((E @ Hd) * E).sum() / den).sqrt())
