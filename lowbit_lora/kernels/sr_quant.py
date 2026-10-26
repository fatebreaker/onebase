"""Stochastic-rounding FP8 (e4m3) activation quantization for an unbiased low-precision sampler (ICML plan v2: the SR kernel;
ICML-E1B-TF measured the sampler it makes: bias 0.22x round-to-nearest's KL). Per token (row), vLLM's scale rule
s = max(amax / 448, 1 / (448 * 512)); then y = x / s rounded to one of its two e4m3 neighbours, the upper one with
probability (y - lo) / (hi - lo), so that E[q] = y exactly (no bias) and |q - y| < one e4m3 step. The neighbours come from the
round-to-nearest code (PTX cvt.rn on sm_89+, as K1) stepped by one code toward y; the random numbers from Philox
(tl.rand(seed, offset)), one per element, so a seed and the element's flat index fix the outcome (reproducible) and a new
seed per forward gives fresh noise. Writes the codes and the per-token scales, as K1's all-token pass does, so the plugin's
hybridq path can feed them to the vendor FP8 GEMM (an SR sampler at W8A8 speed).
    from lowbit_lora.kernels.sr_quant import quant_rows_sr
    xq, sx = quant_rows_sr(x, seed)            # x [M, K] bf16 -> xq [M, K] float8_e4m3fn, sx [M] fp32
CPU check: TRITON_INTERPRET=1 python tests/sr_quant_interp_check.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
_MIN_SCALE = 1.0 / (448.0 * 512.0)


@triton.jit
def _e4m3_value(mag, neg):
    """fp32 value of the e4m3 code with magnitude bits mag (0..0x7E) and sign neg."""
    code = (mag | tl.where(neg, 0x80, 0)).to(tl.uint8)
    return code.to(tl.float8e4nv, bitcast=True).to(tl.float32)


@triton.jit
def _sr_quant_rows(X, XQ, SX, M, K, stride_xm, stride_qm, seed,
                   BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, RN_ASM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mrow = rows < M
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=mrow[:, None] & (offs_k[None, :] < K),
                    other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
    sx = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
    tl.store(SX + rows, sx, mask=mrow)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        msk = mrow[:, None] & (offs_k[None, :] < K)
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=msk, other=0.0)
        y = tl.clamp(x.to(tl.float32) / sx[:, None], -448.0, 448.0)
        if RN_ASM:   # GPU: the same round-to-nearest conversion as K1
            q = tl.inline_asm_elementwise(asm="cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;", constraints="=h,r,r", args=[y],
                                          dtype=tl.float8e4nv, is_pure=True, pack=2)
        else:        # CPU interpreter: its round to nearest drops the carry into the exponent (124.3 -> 64), its rtz is exact;
            q = y.to(tl.float8e4nv, fp_downcast_rounding="rtz")     # either neighbour serves, the next lines find the other
        qf = q.to(tl.float32)
        code = q.to(tl.uint8, bitcast=True).to(tl.int32)
        neg = (code & 0x80) != 0
        mag = code & 0x7F
        up_mag = tl.minimum(mag + 1, 0x7E)                   # 0x7E = 448; 0x7F is NaN
        dn_mag = tl.maximum(mag - 1, 0)
        up = tl.where(neg, _e4m3_value(dn_mag, neg), _e4m3_value(up_mag, neg))      # toward +inf
        dn = tl.where(neg, _e4m3_value(up_mag, neg), _e4m3_value(dn_mag, neg))      # toward -inf
        zneg = neg & (mag == 0)                              # -0 steps up to the smallest positive code
        up = tl.where(zneg, _e4m3_value(tl.full(mag.shape, 1, tl.int32), tl.zeros(neg.shape, tl.int1)), up)
        lo = tl.where(qf <= y, qf, dn)
        hi = tl.where(qf <= y, tl.where(qf == y, qf, up), qf)
        gap = hi - lo
        p = tl.where(gap > 0, (y - lo) / tl.where(gap > 0, gap, 1.0), 0.0)
        u = tl.rand(seed, rows[:, None] * K + offs_k[None, :])
        r = tl.where(u < p, hi, lo)
        tl.store(XQ + rows[:, None].to(tl.int64) * stride_qm + offs_k[None, :], r.to(tl.float8e4nv), mask=msk)


def quant_rows_sr(x: torch.Tensor, seed: int, out: torch.Tensor | None = None, scales: torch.Tensor | None = None,
                  block_m: int = 16, block_k: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Stochastic-rounding per-token e4m3 quantization of x [M, K] (bf16 or fp32): (codes [M, K] float8_e4m3fn, scales [M])."""
    M, K = x.shape
    xq = out if out is not None else torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    sx = scales if scales is not None else torch.empty(M, dtype=torch.float32, device=x.device)
    _sr_quant_rows[(triton.cdiv(M, block_m),)](x, xq, sx, M, K, x.stride(0), xq.stride(0), seed, BLOCK_M=block_m,
                                               BLOCK_K=block_k, RN_ASM=x.is_cuda)
    return xq, sx
