"""Swap-AB weight-only FP8 (W8A16) GEMM for Hopper (MATCHED-HOPPER-KERNEL, docs/REGISTRY.md): the matched engine's base GEMM
(w8a16_lora_multi.w8a16_linear) with the operands swapped,

    y^T[n, m] = s_w * sum_k widen(W_q)[n, k] * x[m, k]

so the e4m3 weight codes, read as bytes and widened to bf16 in registers (exact), are wgmma's register (left) operand and the
bf16 activations its shared-memory (right) operand. Tokens become wgmma's N dimension (16 and up), so a decode batch of 1-64
tokens still runs on Hopper's warpgroup MMA instead of the m16n8k16 mma.sync the unswapped kernel falls back to at BLOCK_M 16
(sm_90 PTX: tools in the registry entry), and the widened weights never take a round trip through shared memory. Arithmetic is
the unswapped kernel's: bf16 products accumulated in fp32 over K, the per-tensor scale once at the end, one rounding to bf16;
only the order of the fp32 additions inside an MMA differs. Optional split-K (SPLIT_K > 1) for small batches on narrow
projections: fp32 partials in a workspace, summed in a fixed order by a second kernel (deterministic).

    from lowbit_lora.kernels.w8a16_swap import w8a16_linear_swap
    y = w8a16_linear_swap(x, w_q, s_w)        # x [M, K] bf16, w_q [N, K] float8_e4m3fn, s_w [1] fp32 -> [M, N] bf16
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w16_swap(X, W, SW, Y, P, M, N, K, K_SPLIT, stride_xm, stride_wn, stride_ym,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
              SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, DOT_F32: tl.constexpr = False):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_N * num_m
    first_n = (pid // group) * GROUP_N
    gsz = tl.minimum(num_n - first_n, GROUP_N)
    pid_n = first_n + (pid % group) % gsz
    pid_m = (pid % group) // gsz
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    mn = offs_n < N
    k_lo = pid_k * K_SPLIT
    k_hi = tl.minimum(k_lo + K_SPLIT, K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mn[:, None], other=0.0)
            xt = tl.load(X + offs_m[None, :] * stride_xm + offs_k[:, None], mask=mm[None, :], other=0.0)
        else:
            w = tl.load(W + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mn[:, None] & (offs_k[None, :] < k_hi),
                        other=0.0)
            xt = tl.load(X + offs_m[None, :] * stride_xm + offs_k[:, None], mask=mm[None, :] & (offs_k[:, None] < k_hi),
                         other=0.0)
        if DOT_F32:   # CPU interpreter only: its bf16 tl.dot is wrong
            acc = tl.dot(w.to(tl.bfloat16).to(tl.float32), xt.to(tl.float32), acc)
        else:
            acc = tl.dot(w.to(tl.bfloat16), xt, acc)       # codes widened exactly; fp32 accumulation
    if SPLIT_K == 1:
        acc = acc * tl.load(SW)
        tl.store(Y + offs_m[None, :] * stride_ym + offs_n[:, None], acc.to(tl.bfloat16), mask=mm[None, :] & mn[:, None])
    else:                                                   # fp32 partial [SPLIT_K, M, N]
        tl.store(P + pid_k * M * N + offs_m[None, :] * N + offs_n[:, None], acc, mask=mm[None, :] & mn[:, None])


@triton.jit
def _splitk_reduce(P, SW, Y, M, N, stride_ym, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    msk = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(SPLIT_K):                      # fixed order: deterministic
        acc += tl.load(P + j * M * N + offs, mask=msk, other=0.0)
    acc = acc * tl.load(SW)
    m = offs // N
    n = offs % N
    tl.store(Y + m * stride_ym + n, acc.to(tl.bfloat16), mask=msk)


def swap_cfg(M: int, N: int, K: int) -> dict:
    """Default tiles by batch size (placeholder until the H100 sweep; bench/w16_swap_bench.py --tune)."""
    bm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    return {"BLOCK_M": bm, "BLOCK_N": 128, "BLOCK_K": 128 if bm <= 64 else 64, "SPLIT_K": 1, "num_warps": 4,
            "num_stages": 4 if bm <= 64 else 3}


_WS: dict = {}   # (device, numel) -> fp32 split-K workspace (static, so CUDA graphs can capture the call)


def w8a16_linear_swap(x: torch.Tensor, w_q: torch.Tensor, s_w: torch.Tensor, cfg: dict | None = None,
                      dot_f32: bool = False) -> torch.Tensor:
    """y = x @ (s_w * widen(w_q))^T: x [M, K] bf16 row-major, w_q [N, K] float8_e4m3fn (row-major), s_w [1] fp32.
    Graph-capturable (no host synchronization)."""
    M, K = x.shape
    N = w_q.shape[0]
    c = cfg or swap_cfg(M, N, K)
    sk = c.get("SPLIT_K", 1)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    k_split = triton.cdiv(triton.cdiv(K, sk), c["BLOCK_K"]) * c["BLOCK_K"]
    if sk > 1:
        key = (x.device, sk * M * N)
        p = _WS.get(key)
        if p is None:
            p = _WS[key] = torch.empty(sk * M * N, dtype=torch.float32, device=x.device)
    else:
        p = y
    grid = (triton.cdiv(M, c["BLOCK_M"]) * triton.cdiv(N, c["BLOCK_N"]), sk)
    _w16_swap[grid](x, w_q, s_w, y, p, M, N, K, k_split, x.stride(0), w_q.stride(0), y.stride(0),
                    BLOCK_M=c["BLOCK_M"], BLOCK_N=c["BLOCK_N"], BLOCK_K=c["BLOCK_K"], GROUP_N=c.get("GROUP_N", 8),
                    SPLIT_K=sk, EVEN_K=(K % c["BLOCK_K"] == 0 and k_split % c["BLOCK_K"] == 0), DOT_F32=dot_f32,
                    num_warps=c["num_warps"], num_stages=c["num_stages"])
    if sk > 1:
        blk = 1024
        _splitk_reduce[(triton.cdiv(M * N, blk),)](p, s_w, y, M, N, y.stride(0), SPLIT_K=sk, BLOCK=blk, num_warps=4)
    return y
