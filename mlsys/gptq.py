"""Minimal GPTQ (Frantar et al., 2023) for symmetric INT-k with groups along the input dimension.

    Q, scales = gptq(W, H, bits=4, group=128)      # W [out, in] float32; H = X^T X over calibration inputs [in, in]

Columns are quantized in order; each column's rounding error is fed back into the not-yet-quantized columns through
the upper Cholesky factor of H^-1, minimizing ||(W - Q) X^T|| on the calibration inputs. Group scales are absmax of
the (already error-updated) weights of each group of 128 columns, as in GPTQ with static groups computed on the fly.
Used for the remedies of the paper: generic vs on-policy calibration data (mlsys.acts captures the inputs).
"""
from __future__ import annotations

import torch


def hessian(X: torch.Tensor) -> torch.Tensor:
    """X [n_tokens, in] -> X^T X / n (float64 accumulation)."""
    X = X.double()
    return (X.T @ X / X.shape[0]).float()


def gptq(W: torch.Tensor, H: torch.Tensor, bits: int = 4, group: int = 128, damp: float = 0.01,
         block: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if block % group and group % block:
        raise ValueError("block and group must nest")
    W = W.clone().float()
    n_out, n_in = W.shape
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(n_in, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    Q = torch.zeros_like(W)
    scales = torch.zeros(n_out, (n_in + group - 1) // group, device=W.device)
    scale = None
    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        H1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:                           # scale from the error-updated weights of this group
                gw = torch.cat([W1[:, i:], W[:, i2:]], dim=1)[:, :group]
                scale = (gw.abs().amax(1) / qmax).clamp_min(1e-12)
                scales[:, col // group] = scale
            w = W1[:, i]
            q = torch.clamp(torch.round(w / scale), -qmax, qmax) * scale
            Q1[:, i] = q
            err = (w - q) / H1[i, i]
            W1[:, i:] -= err[:, None] * H1[i, i:][None, :]
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, scales
