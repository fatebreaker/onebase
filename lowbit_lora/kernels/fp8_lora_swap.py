"""Swap-AB FP8 GEMM with the multi-adapter LoRA epilogue, for Hopper decode (the plugin's fused path, K2, on sm_90).

fp8_lora_tma._gemm_tma runs K2 on 64-row WGMMA tiles over the tokens in their natural order. A decode batch of 1-16 tokens
fills a quarter or less of such a tile, and with 64-wide N tiles a projection launches about N / 64 CTAs, fewer than the
H100's 132 SMs on Qwen3-8B's o and down projections, which then stream their weights through half the GPU (the down
projection at 1.15-1.47x the hybrid path's time; MLSys draft, supplement). As w8a16_swap.py does for the matched engine,
this kernel swaps the operands,

    y^T[n, m] = s_x[m] s_w sum_k W_q[n, k] x_q[m, k]  +  sum_l [token m on adapter l] (B_l h_m^T)[n]

so the weight rows fill wgmma's 64-row side and the tokens its N side (16 and up), and it splits K across CTAs for small
batches: fp32 partials in a static workspace, summed in a fixed order by a second kernel that also applies the scales
(deterministic). The LoRA expansion is added once, by split 0, divided by the token's scale product s_x[m] s_w, which the
reduction's scaling restores (fp32: a few ulp). x_q, s_x and the shrink partials h come from K1 (_shrink_vllm) exactly as
for the other fused paths, so the arithmetic is the fused path's: e4m3 products accumulated in fp32 (promoted after every
K-block with PRECISE, as CUTLASS does), the scales once, the expansion as bf16 products accumulated in fp32, one rounding to
bf16; only the order of fp32 additions differs.

Not wired into the plugin: lowbit_lora/hopper_swap.py patches VllmFusedFP8LoRA when LOWBIT_SWAP=1 is set (a separate module,
so the registered runs' code paths are unchanged).
    TRITON_INTERPRET=1 python tests/swap_k2_interp_check.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile


@triton.jit
def _gemm_swap(XQ, SX, W, SW, HP, B_PTRS, SB_PTRS, BG, Y, P, TOK, SEG_IDS, SEG_N, N_SEG,
               M, N, K, R, S, END0, END1, END2, K_SPLIT, stride_xqm, stride_wn, stride_ym,
               SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
               SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, PRECISE: tl.constexpr,
               B_FP8: tl.constexpr = False, B_FP4: tl.constexpr = False, DOT_F32: tl.constexpr = False):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_N * num_m
    first_n = (pid // group) * GROUP_N
    gsz = tl.minimum(num_n - first_n, GROUP_N)
    pid_n = first_n + (pid % group) % gsz
    pid_m = (pid % group) // gsz
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    mn = offs_n < N
    k_lo = pid_k * K_SPLIT
    k_hi = tl.minimum(k_lo + K_SPLIT, K)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)          # y^T tile: weight rows x tokens
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            w = tl.load(W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=mn[:, None], other=0.0)
            xt = tl.load(XQ + offs_m[None, :].to(tl.int64) * stride_xqm + offs_k[:, None], mask=mm[None, :], other=0.0)
        else:
            mk = offs_k < k_hi
            w = tl.load(W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=mn[:, None] & mk[None, :],
                        other=0.0)
            xt = tl.load(XQ + offs_m[None, :].to(tl.int64) * stride_xqm + offs_k[:, None], mask=mm[None, :] & mk[:, None],
                         other=0.0)
        if DOT_F32:   # CPU interpreter only
            acc = tl.dot(w.to(tl.float32), xt.to(tl.float32), acc)
        elif PRECISE:   # FP8 MMA accumulates with reduced precision: promote to fp32 after every K-block, as CUTLASS does
            acc += tl.dot(w, xt)
        else:
            acc = tl.dot(w, xt, acc)
    sx = tl.load(SX + offs_m, mask=mm, other=1.0)
    scale = sx[None, :] * tl.load(SW)                               # [1, BLOCK_M]
    # adapters. Under CUDA graphs the batch is padded to a captured size; vLLM's metadata covers the n_all real tokens
    # only (the segments' total), so columns past n_all count as tokens without an adapter (as fp8_lora_tma).
    n_seg = tl.load(N_SEG)
    n_ad = 0
    n_all = 0
    for i in range(0, n_seg):
        c = tl.load(SEG_N + i)
        n_all += c
        n_ad += tl.where(tl.load(SEG_IDS + i) >= 0, c, 0)
    tok = tl.load(TOK + offs_m, mask=mm & (offs_m < n_all) & (n_ad > 0), other=-1)
    lo = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    if (tl.max(tok, axis=0) >= 0) & (pid_k == 0):
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32) + (n0 >= END2).to(tl.int32)   # up to 4 slices
        n_s0 = tl.where(s == 0, 0, tl.where(s == 1, END0, tl.where(s == 2, END1, END2)))
        n_s = tl.where(s == 0, END0, tl.where(s == 1, END1 - END0, tl.where(s == 2, END2 - END1, N - END2)))
        if B_FP8:
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.float8e4nv))
            sb_base = tl.load(SB_PTRS + s).to(tl.pointer_type(tl.float32))
        elif B_FP4:
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.uint8))
            bs_base = tl.load(SB_PTRS + s).to(tl.pointer_type(tl.float8e4nv))
        else:
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.bfloat16))
        hm = mm & (tok >= 0)
        RC: tl.constexpr = min(BLOCK_R, R_CHUNK)
        for rc in tl.static_range(0, BLOCK_R, RC):
            offs_r = rc + tl.arange(0, RC)
            mr = offs_r < R
            h = tl.zeros((BLOCK_M, RC), dtype=tl.float32)
            for j in range(0, SPLIT_K1):
                h += tl.load(HP + j * (S * M * R) + s * (M * R) + offs_m[:, None] * R + offs_r[None, :],
                             mask=hm[:, None] & mr[None, :], other=0.0)
            hb = h.to(tl.bfloat16)
            for i in range(0, n_seg):
                lora = tl.load(SEG_IDS + i)
                if (lora >= 0) & (tl.load(SEG_N + i) > 0):
                    sel = tok == lora
                    if tl.sum(sel.to(tl.int32), axis=0) > 0:
                        ht = tl.trans(tl.where(sel[:, None], hb, 0.0).to(tl.bfloat16))      # [RC, BLOCK_M]
                        if B_FP4:   # NVFP4-stored B ([N_s, R/2] codes along R, [N_s, R/16] block scales, a global)
                            bg = tl.load(BG + s * (n_seg - 1) + lora)
                            offs_rp = rc // 2 + tl.arange(0, RC // 2)
                            offs_rs = rc // 16 + tl.arange(0, RC // 16)
                            bp = tl.load(b_base + lora.to(tl.int64) * n_s * (R // 2) + (offs_n[:, None] - n_s0) * (R // 2)
                                         + offs_rp[None, :], mask=mn[:, None] & (offs_rp < R // 2)[None, :], other=0)
                            bsc = tl.load(bs_base + lora.to(tl.int64) * n_s * (R // 16) + (offs_n[:, None] - n_s0) * (R // 16)
                                          + offs_rs[None, :], mask=mn[:, None] & (offs_rs < R // 16)[None, :], other=0.0)
                            b = (_dequant_tile(bp, bsc, BLOCK_N, RC) * bg).to(tl.bfloat16)        # [BLOCK_N, RC]
                        else:
                            b = tl.load(b_base + lora.to(tl.int64) * n_s * R + (offs_n[:, None] - n_s0) * R + offs_r[None, :],
                                        mask=mn[:, None] & mr[None, :], other=0.0)
                            if B_FP8:
                                sbv = tl.load(sb_base + lora.to(tl.int64) * n_s + (offs_n - n_s0), mask=mn, other=0.0)
                                b = (b.to(tl.float32) * sbv[:, None]).to(tl.bfloat16)
                        if DOT_F32:
                            lo += tl.dot(b.to(tl.float32), ht.to(tl.float32))
                        else:
                            lo = tl.dot(b, ht, lo)
    if SPLIT_K == 1:
        out = acc * scale + lo
        tl.store(Y + offs_m[None, :].to(tl.int64) * stride_ym + offs_n[:, None], out.to(tl.bfloat16),
                 mask=mm[None, :] & mn[:, None])
    else:                                                           # fp32 partial [SPLIT_K, M, N]; split 0 carries the LoRA
        acc += lo / scale
        tl.store(P + pid_k * M * N + offs_m[None, :].to(tl.int64) * N + offs_n[:, None], acc, mask=mm[None, :] & mn[:, None])


@triton.jit
def _splitk_reduce_sx(P, SX, SW, Y, M, N, stride_ym, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    msk = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for j in tl.static_range(SPLIT_K):                              # fixed order: deterministic
        acc += tl.load(P + j * M * N + offs, mask=msk, other=0.0)
    m = offs // N
    n = offs % N
    acc = acc * (tl.load(SX + m, mask=msk, other=1.0) * tl.load(SW))
    tl.store(Y + m.to(tl.int64) * stride_ym + n, acc.to(tl.bfloat16), mask=msk)


def swap_k2_cfg(M: int, widths) -> dict:
    """Default tiles (placeholders until the H100 sweep: bench/swap_k2_bench.py --tune)."""
    bm = 16 if M <= 16 else 32 if M <= 32 else 64 if M <= 64 else 128
    cg = {"BLOCK_M": bm, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 4 if M <= 16 else 2 if M <= 64 else 1,
          "num_warps": 4, "num_stages": 4, "GROUP_N": 8}
    while any(w % cg["BLOCK_N"] for w in widths) and cg["BLOCK_N"] > 16:
        cg["BLOCK_N"] //= 2
    return cg


_WS: dict = {}   # (device, numel) -> fp32 split-K workspace (static, so CUDA graphs can capture the call)


def workspace(device, numel: int) -> torch.Tensor:
    p = _WS.get((device, numel))
    if p is None:
        p = _WS[(device, numel)] = torch.empty(numel, dtype=torch.float32, device=device)
    return p


def gemm_swap(op, xq, sxb, hp, y, tok, seg_ids, seg_n, split: int, cfg: dict | None = None, precise: bool = True):
    """y = FP8 GEMM(x_q, op.w_q) * scales + every adapter's expansion, with the operands swapped (op: a VllmFusedFP8LoRA or
    anything with its fields). x_q [M, K] e4m3 (natural order), the per-token scales sxb [>= M] and the shrink partials hp
    ([split, S, M, R] fp32, flat) come from _shrink_vllm. Graph-capturable (no host synchronization)."""
    M, K = xq.shape
    N = op.N
    c = dict(cfg or swap_k2_cfg(M, op.widths))
    assert all(w % c["BLOCK_N"] == 0 for w in op.widths), (c["BLOCK_N"], op.widths)
    sk = c.get("SPLIT_K", 1)
    k_split = triton.cdiv(triton.cdiv(K, sk), c["BLOCK_K"]) * c["BLOCK_K"]
    p = workspace(xq.device, sk * M * N) if sk > 1 else y
    br = max(16, triton.next_power_of_2(op.R))
    rck = 64 if op.lora_fp4 else br
    grid = (triton.cdiv(M, c["BLOCK_M"]) * triton.cdiv(N, c["BLOCK_N"]), sk)
    _gemm_swap[grid](
        xq, sxb, op.w_q, op.s_w, hp, op.b_ptrs, op.sb_ptrs, op.bg, y, p, tok, seg_ids, seg_n, op.n_seg,
        M, N, K, op.R, op.S, op.end0, op.end1, op.end2, k_split, xq.stride(0), op.w_q.stride(0), y.stride(0),
        SPLIT_K1=split, BLOCK_R=br, R_CHUNK=rck, BLOCK_M=c["BLOCK_M"], BLOCK_N=c["BLOCK_N"], BLOCK_K=c["BLOCK_K"],
        GROUP_N=c.get("GROUP_N", 8), SPLIT_K=sk, EVEN_K=(K % c["BLOCK_K"] == 0 and k_split % c["BLOCK_K"] == 0),
        PRECISE=precise, B_FP8=op.lora_fp8, B_FP4=op.lora_fp4, DOT_F32=getattr(op, "dot_f32", False),
        num_warps=c["num_warps"], num_stages=c["num_stages"])
    if sk > 1:
        blk = 1024
        _splitk_reduce_sx[(triton.cdiv(M * N, blk),)](p, sxb, op.s_w, y, M, N, y.stride(0), SPLIT_K=sk, BLOCK=blk,
                                                     num_warps=4)
    return y
