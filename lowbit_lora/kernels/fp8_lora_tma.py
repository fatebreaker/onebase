"""Hopper (sm_90) FP8 GEMM with the multi-adapter LoRA epilogue, fed by TMA (MLSys v2, docs/PLAN_MLSYS_V2.md).

fp8_lora_vllm._gemm_vllm runs its GEMM over vLLM's adapter-sorted token order, gathering x_q rows with pointer loads in
16-row tiles. That suits Ada (mma.sync) but on Hopper it rules out TMA and the 64-row WGMMA tiles, and CUTLASS's FP8 GEMM
wins (the timed dispatch then keeps the hybrid path: CUTLASS, then our LoRA kernels). Here the GEMM runs over the tokens
in their natural order: K1 (_shrink_vllm) already writes x_q and the per-token scales in natural order, and it stores each
token's shrink h at the token's own row, so the base GEMM streams x_q and W tiles through TMA descriptors into WGMMA, and
the epilogue adds, for every adapter segment present in the tile, that adapter's expansion h_l B_l^T with the rows of other
adapters masked out (a decode tile meets at most the batch's few dozen adapters; each masked dot is a small tensor-core op,
and each adapter's B slice is read once per N-tile, as any exact path must read it).

Opt-in: LOWBIT_TMA=1 on a compute-capability-9 GPU (VllmFusedFP8LoRA._run); off by default, so registered runs are unchanged.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile


@triton.jit
def _gemm_tma(xq_desc, w_desc, SX, SW, HP, B_PTRS, SB_PTRS, BG, Y, TOK, SEG_IDS, SEG_N, N_SEG,
              M, N, K, R, S, END0, END1, END2, stride_ym,
              SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
              NUM_STAGES: tl.constexpr, WS: tl.constexpr, PRECISE: tl.constexpr,
              B_FP8: tl.constexpr = False, B_FP4: tl.constexpr = False, DOT_F32: tl.constexpr = False):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsz
    pid_n = (pid % group) // gsz
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES, warp_specialize=WS):
        a = xq_desc.load([m0, k])
        b = w_desc.load([n0, k])
        if PRECISE:   # FP8 MMA accumulates with reduced precision: promote to fp32 after every K-block, as CUTLASS does
            acc += tl.dot(a, b.T)
        else:
            acc = tl.dot(a, b.T, acc)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mm = offs_m < M
    mn = offs_n < N
    sx = tl.load(SX + offs_m, mask=mm, other=1.0)
    acc = acc * (sx[:, None] * tl.load(SW))
    # adapters. Under CUDA graphs the batch is padded to a captured size; vLLM's metadata covers the n_all real tokens
    # only (the segments' total), so rows past n_all count as tokens without an adapter.
    n_seg = tl.load(N_SEG)
    n_ad = 0
    n_all = 0
    for i in range(0, n_seg):
        c = tl.load(SEG_N + i)
        n_all += c
        n_ad += tl.where(tl.load(SEG_IDS + i) >= 0, c, 0)
    tok = tl.load(TOK + offs_m, mask=mm & (offs_m < n_all) & (n_ad > 0), other=-1)
    if tl.max(tok, axis=0) >= 0:
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
                        h_l = tl.where(sel[:, None], hb, 0.0).to(tl.bfloat16)
                        if B_FP4:   # NVFP4-stored B ([N_s, R/2] codes along R, [N_s, R/16] block scales, a global)
                            bg = tl.load(BG + s * (n_seg - 1) + lora)
                            offs_rp = rc // 2 + tl.arange(0, RC // 2)
                            offs_rs = rc // 16 + tl.arange(0, RC // 16)
                            bp = tl.load(b_base + lora.to(tl.int64) * n_s * (R // 2) + (offs_n[:, None] - n_s0) * (R // 2)
                                         + offs_rp[None, :], mask=mn[:, None] & (offs_rp < R // 2)[None, :], other=0)
                            bsc = tl.load(bs_base + lora.to(tl.int64) * n_s * (R // 16) + (offs_n[:, None] - n_s0) * (R // 16)
                                          + offs_rs[None, :], mask=mn[:, None] & (offs_rs < R // 16)[None, :], other=0.0)
                            bt = tl.trans(_dequant_tile(bp, bsc, BLOCK_N, RC) * bg).to(tl.bfloat16)
                        else:
                            bt = tl.load(b_base + lora.to(tl.int64) * n_s * R + (offs_n[None, :] - n_s0) * R + offs_r[:, None],
                                         mask=mn[None, :] & mr[:, None], other=0.0)
                            if B_FP8:
                                sbv = tl.load(sb_base + lora.to(tl.int64) * n_s + (offs_n - n_s0), mask=mn, other=0.0)
                                bt = (bt.to(tl.float32) * sbv[None, :]).to(tl.bfloat16)
                        if DOT_F32:   # tests (Triton's CPU interpreter computes bf16 dots wrongly)
                            acc += tl.dot(h_l.to(tl.float32), bt.to(tl.float32))
                        else:
                            acc = tl.dot(h_l, bt, acc)
    tl.store(Y + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :], acc.to(tl.bfloat16),
             mask=mm[:, None] & mn[None, :])


def pick_cfg_tma(M: int, widths) -> dict:
    """Default tiles (to be tuned on the H100: bench/hopper_tma_bench.py): 64-row WGMMA tiles for decode, 128x128 above."""
    if M <= 64:
        cg = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "NUM_STAGES": 4, "WS": False}
    elif M <= 256:
        cg = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "NUM_STAGES": 4, "WS": False}
    else:
        cg = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "NUM_STAGES": 3, "WS": False}
    while any(w % cg["BLOCK_N"] for w in widths) and cg["BLOCK_N"] > 32:
        cg["BLOCK_N"] //= 2
    cg["GROUP_M"] = 8
    return cg


def gemm_tma(op, x, xq, sxb, hp, y, tok, seg_ids, seg_n, split: int, cfg: dict | None = None, precise: bool = True):
    """y = FP8 GEMM(x_q in natural order, op.w_q) * scales + every adapter's expansion (op: a VllmFusedFP8LoRA). x_q, the
    per-token scales and the shrink partials hp come from _shrink_vllm (run before, as in the fused path)."""
    M = x.shape[0]
    cg = dict(cfg or pick_cfg_tma(M, op.widths))
    bm, bn, bk = cg["BLOCK_M"], cg["BLOCK_N"], cg["BLOCK_K"]
    xq_desc = TensorDescriptor.from_tensor(xq, [bm, bk])
    w_key = (bn, bk)
    if getattr(op, "_w_desc", None) is None or op._w_desc[0] != w_key:   # W never changes: one descriptor per tile shape
        op._w_desc = (w_key, TensorDescriptor.from_tensor(op.w_q, [bn, bk]))
    w_desc = op._w_desc[1]
    br = max(16, triton.next_power_of_2(op.R))
    rck = 64 if op.lora_fp4 else br
    grid = (triton.cdiv(M, bm) * triton.cdiv(op.N, bn),)
    _gemm_tma[grid](
        xq_desc, w_desc, sxb, op.s_w, hp, op.b_ptrs, op.sb_ptrs, op.bg, y, tok, seg_ids, seg_n, op.n_seg,
        M, op.N, op.K, op.R, op.S, op.end0, op.end1, op.end2, y.stride(0),
        SPLIT_K1=split, BLOCK_R=br, R_CHUNK=rck, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=cg["GROUP_M"],
        NUM_STAGES=cg["NUM_STAGES"], WS=cg["WS"], PRECISE=precise, B_FP8=op.lora_fp8, B_FP4=op.lora_fp4,
        DOT_F32=getattr(op, "dot_f32", False), num_warps=cg["num_warps"])
    return y
