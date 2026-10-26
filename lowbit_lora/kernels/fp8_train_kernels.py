"""Training-side FP8 GEMM for the colocated-RL trainer (RL-KERNELS): K2's arithmetic (per-row FP8 activations or gradients,
the engine's per-tensor-scaled e4m3 weight codes, fp32 accumulation promoted per K-block, an optional LoRA expand in the
epilogue) with the weight operand read through two strides, so the backward reads W's codes in place:

    forward   y  = s_x s_W (Q(x) W^T) [+ bf16(h) B^T]      W [N, K] read with (stride_n, stride_k) = (K, 1)
    backward  dX = s_g s_W (Q(g) W)   [+ bf16(u) A]        the same W read as [K_out, N_red] with (1, K)

instead of transposing the codes on every backward call (GEMM-BENCH, L40S: the transpose of a 12288 x 4096 weight took
0.58 ms against 0.40 ms for the FP8 GEMM itself). Configurations are autotuned per (N, K, layout), not per M, since a
trainer's M changes with every micro-batch. The serving kernels (fp8_lora.py) are left unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = tl.constexpr(448.0)


def _configs():
    out = []
    for bm, bn, bk, nw, ns in [(128, 128, 128, 8, 3), (128, 128, 64, 4, 4), (128, 256, 64, 8, 3), (64, 128, 128, 4, 4),
                               (128, 64, 128, 4, 4), (256, 128, 64, 8, 3)]:
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=nw, num_stages=ns))
    return out


@triton.autotune(configs=_configs(), key=["N", "K", "W_KMAJOR", "HAS_LORA"])
@triton.jit
def _gemm_train_kernel(XQ, SX, W, SW, HP, B, Y, M, N, K, R,
                       stride_xm, stride_wn, stride_wk, stride_bn, stride_ym,
                       W_KMAJOR: tl.constexpr, HAS_LORA: tl.constexpr, BLOCK_R: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz
    pid_n = (pid % group) // gsz
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    mn = offs_n < N
    sx = tl.load(SX + offs_m, mask=mm, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        a = tl.load(XQ + offs_m[:, None] * stride_xm + offs_k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if W_KMAJOR:     # forward: W [N, K], K contiguous
            b = tl.load(W + offs_n[None, :] * stride_wn + offs_k[:, None], mask=mn[None, :] & mk[:, None], other=0.0)
        else:            # backward: W's codes in place, the reduction runs over its rows
            b = tl.load(W + offs_n[None, :] + offs_k[:, None] * stride_wk, mask=mn[None, :] & mk[:, None], other=0.0)
        acc += tl.dot(a, b)            # promoted to fp32 after every K-block, as K2 with PRECISE
    acc = acc * (sx[:, None] * tl.load(SW))
    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)
        mr = offs_r < R
        h = tl.load(HP + offs_m[:, None] * R + offs_r[None, :], mask=mm[:, None] & mr[None, :], other=0.0)
        bt = tl.load(B + offs_n[None, :] * stride_bn + offs_r[:, None], mask=mn[None, :] & mr[:, None], other=0.0)
        acc = tl.dot(h.to(tl.bfloat16), bt, acc)
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


def gemm(xq: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor, transposed: bool = False,
         hp: torch.Tensor | None = None, b: torch.Tensor | None = None) -> torch.Tensor:
    """xq [M, Kr] e4m3 (row-major), sx [M] fp32; w [N0, K0] e4m3 row-major with the per-tensor scale sw.
    transposed=False: y = xq @ w^T, [M, N0] (Kr = K0). transposed=True: y = xq @ w, [M, K0] (Kr = N0), w read in place.
    With hp [M, r] fp32 and b [Nout, r] bf16: + bf16(hp) @ b^T in the epilogue."""
    M = xq.shape[0]
    n0, k0 = w.shape
    if transposed:
        N, K, swn, swk = k0, n0, 1, k0
    else:
        N, K, swn, swk = n0, k0, k0, 1
    assert xq.shape[1] == K
    y = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
    has = hp is not None
    R = b.shape[1] if has else 16
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    _gemm_train_kernel[grid](xq, sx, w, sw, hp if has else sx, b if has else w, y, M, N, K, R,
                             xq.stride(0), swn, swk, b.stride(0) if has else 0, y.stride(0),
                             W_KMAJOR=not transposed, HAS_LORA=has, BLOCK_R=max(16, triton.next_power_of_2(R)))
    return y


@triton.autotune(configs=_configs(), key=["N", "K", "W_KMAJOR", "HAS_LORA"])
@triton.jit
def _gemm_w8a16_kernel(X, W, SW, HP, B, Y, M, N, K, R,
                       stride_xm, stride_wn, stride_wk, stride_bn, stride_ym,
                       W_KMAJOR: tl.constexpr, HAS_LORA: tl.constexpr, BLOCK_R: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """W8A16: bf16 activations, the engine's e4m3 codes widened to bf16 in registers (exact), bf16 MMA with fp32
    accumulation, the per-tensor scale once at the end; the weight-matched trainer's arithmetic without materializing W."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz
    pid_n = (pid % group) // gsz
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    mn = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < K
        a = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if W_KMAJOR:
            b = tl.load(W + offs_n[None, :] * stride_wn + offs_k[:, None], mask=mn[None, :] & mk[:, None], other=0.0)
        else:
            b = tl.load(W + offs_n[None, :] + offs_k[:, None] * stride_wk, mask=mn[None, :] & mk[:, None], other=0.0)
        acc = tl.dot(a, b.to(tl.bfloat16), acc)
    acc = acc * tl.load(SW)
    if HAS_LORA:
        offs_r = tl.arange(0, BLOCK_R)
        mr = offs_r < R
        h = tl.load(HP + offs_m[:, None] * R + offs_r[None, :], mask=mm[:, None] & mr[None, :], other=0.0)
        bt = tl.load(B + offs_n[None, :] * stride_bn + offs_r[:, None], mask=mn[None, :] & mr[:, None], other=0.0)
        acc = tl.dot(h.to(tl.bfloat16), bt, acc)
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


def gemm_w8a16(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor, transposed: bool = False,
               hp: torch.Tensor | None = None, b: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, Kr] bf16 (row-major); w [N0, K0] e4m3 row-major with the per-tensor scale sw. transposed=False: x @ (sw w)^T,
    [M, N0]; transposed=True: x @ (sw w), [M, K0], w read in place. hp [M, r] (fp32 or bf16) and b [Nout, r] bf16: + bf16(hp) @ b^T."""
    M = x.shape[0]
    n0, k0 = w.shape
    if transposed:
        N, K, swn, swk = k0, n0, 1, k0
    else:
        N, K, swn, swk = n0, k0, k0, 1
    assert x.shape[1] == K and x.dtype == torch.bfloat16 and x.stride(1) == 1
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    has = hp is not None
    R = b.shape[1] if has else 16
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)  # noqa: E731
    _gemm_w8a16_kernel[grid](x, w, sw, hp if has else x, b if has else x, y, M, N, K, R,
                             x.stride(0), swn, swk, b.stride(0) if has else 0, y.stride(0),
                             W_KMAJOR=not transposed, HAS_LORA=has, BLOCK_R=max(16, triton.next_power_of_2(R)))
    return y
