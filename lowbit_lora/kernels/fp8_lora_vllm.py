"""Fused FP8 GEMM + multi-adapter LoRA on vLLM's own LoRA buffers (Phase 1; the kernels of fp8_lora_multi.py, adapted).

Differences from fp8_lora_multi.py, so that a vLLM LoRA layer can call it with no copies:
  * adapter weights are vLLM's per-slice stacked buffers, lora_a_stacked[s] [L, 1, R, K] and lora_b_stacked[s]
    [L, 1, N_s, R] (bf16), reached through device pointer tables (as vLLM's Punica kernels do);
  * token metadata is vLLM's LoRAKernelMeta: token_lora_mapping, token_indices_sorted_by_lora_ids,
    num_tokens_per_lora, lora_token_start_loc, active_lora_ids (L + 1 segments; -1 = tokens without an adapter);
  * vLLM folds alpha / r into B, so the shrink is unscaled (scale = 1).
A column block of the shrink and an N-tile of the GEMM each lie inside one slice (BLOCK_SR divides R; BLOCK_N divides
every slice width).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.common import shrink_choice, to_e4m3
from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile, _e2m1_value, nvfp4_quant_batched

FP8_MAX = 448.0
SPLIT_M_MAX = 512   # the shrink splits K up to this many tokens: without it, a 256-token decode batch has a few dozen
                    # programs each walking all of K (14336 on Llama's down projection), slower than vLLM's shrink
_FP8_MAX = tl.constexpr(448.0)   # the same constant, usable inside @triton.jit kernels
_MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))   # vLLM's floor on a per-token scale


@triton.jit
def _shrink_vllm(X, A_PTRS, SA_PTRS, AG, HP, PMAX, XQ, SX, SORTED, SEG_IDS, SEG_N, SEG_START,
                 M, K, R, S, L1, stride_xm, stride_al, stride_am, stride_xqm,
                 SPLIT_K: tl.constexpr, WRITE_XQ: tl.constexpr, N_SR: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr, BLOCK_K: tl.constexpr, DOT_F32: tl.constexpr = False,
                 RN: tl.constexpr = True, A_FP8: tl.constexpr = False, A_FP4: tl.constexpr = False):
    """grid axis 2: segments 0..L (vLLM's sorted adapter segments: the shrink) and one more, L1 = L + 1: the pass over
    all tokens in natural order that computes the per-token maxima (and x_q in prefill mode). The maxima never depend
    on the adapter metadata, which vLLM leaves stale when a batch has no adapter."""
    pid = tl.program_id(0)
    seg = tl.program_id(2)
    pid_k = pid % SPLIT_K
    pid_r = (pid // SPLIT_K) % N_SR
    pid_m = pid // (SPLIT_K * N_SR)
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    if seg == L1:   # all tokens, natural order: full-row maxima, scale, and x_q, always (once per layer)
        if pid_m * BLOCK_M >= M:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mrow = rows < M
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):            # every program of this row tile reads the full rows (cheap)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
                        mask=mrow[:, None] & (offs_k[None, :] < K), other=0.0)
            amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
        p = pid_r * SPLIT_K + pid_k                 # the N_SR * SPLIT_K programs of this tile share the x_q pass
        if p == 0:
            tl.store(SX + rows, sx, mask=mrow)
        k_slice = tl.cdiv(tl.cdiv(K, N_SR * SPLIT_K), BLOCK_K) * BLOCK_K
        q_lo = p * k_slice
        q_hi = tl.minimum(q_lo + k_slice, K)
        for k0 in range(q_lo, q_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            msk = mrow[:, None] & (offs_k[None, :] < q_hi)
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=msk, other=0.0)
            q = to_e4m3(tl.clamp(tl.math.div_rn(x.to(tl.float32), sx[:, None]), -_FP8_MAX, _FP8_MAX), RN)
            tl.store(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_k[None, :], q, mask=msk)
        return
    lora = tl.load(SEG_IDS + seg)                                       # an adapter segment: the shrink only
    seg_n = tl.load(SEG_N + seg)
    if lora < 0:
        return
    if pid_m * BLOCK_M >= seg_n:
        return
    seg_start = tl.load(SEG_START + seg)
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mrow = offs < seg_n
    rows = tl.load(SORTED + seg_start + offs, mask=mrow, other=0)
    c0 = pid_r * BLOCK_SR
    s = c0 // R                                                        # the slice of this column block
    offs_c = (c0 - s * R) + tl.arange(0, BLOCK_SR)                      # rank rows within the slice
    if A_FP8:   # FP8-stored adapter (E4M3, one fp32 scale per rank row): exact upcast to bf16, scale after the dot
        a_base = tl.load(A_PTRS + s).to(tl.pointer_type(tl.float8e4nv)) + lora.to(tl.int64) * stride_al
    elif A_FP4:   # NVFP4-stored adapter ([R, K/2] codes, [R, K/16] E4M3 block scales, a global per adapter and slice)
        a_base = tl.load(A_PTRS + s).to(tl.pointer_type(tl.uint8)) + lora.to(tl.int64) * (R * (K // 2))
        as_base = tl.load(SA_PTRS + s).to(tl.pointer_type(tl.float8e4nv)) + lora.to(tl.int64) * (R * (K // 16))
    else:
        a_base = tl.load(A_PTRS + s).to(tl.pointer_type(tl.bfloat16)) + lora.to(tl.int64) * stride_al
    acc = tl.zeros((BLOCK_M, BLOCK_SR), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < k_hi
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=mrow[:, None] & mk[None, :],
                    other=0.0)
        if A_FP4:   # decode to bf16 in registers (code * E4M3 scale is exact in bf16), vectorized loads + join/reshape;
            # safe here because BLOCK_K >= 64 (tests/gpu_dequant_check.py: wrong only for 16-wide tiles fed to a dot)
            offs_kp = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            offs_ks = k0 // 16 + tl.arange(0, BLOCK_K // 16)
            ap = tl.load(a_base + offs_c[:, None] * (K // 2) + offs_kp[None, :], mask=(offs_kp < k_hi // 2)[None, :],
                         other=0)
            asc = tl.load(as_base + offs_c[:, None] * (K // 16) + offs_ks[None, :],
                          mask=(offs_ks < k_hi // 16)[None, :], other=0.0)
            a = _dequant_tile(ap, asc, BLOCK_SR, BLOCK_K).to(tl.bfloat16)
        else:
            a = tl.load(a_base + offs_c[:, None] * stride_am + offs_k[None, :], mask=mk[None, :], other=0.0)
            if A_FP8:
                a = a.to(tl.bfloat16)
        if DOT_F32:
            acc = tl.dot(x.to(tl.float32), tl.trans(a.to(tl.float32)), acc)
        else:
            acc = tl.dot(x, tl.trans(a), acc)
    if A_FP8:
        acc = acc * tl.load(tl.load(SA_PTRS + s).to(tl.pointer_type(tl.float32)) + lora.to(tl.int64) * R + offs_c)[None, :]
    elif A_FP4:
        acc = acc * tl.load(AG + s * (L1 - 1) + lora)
    hp = HP + pid_k * (S * M * R) + s * (M * R) + rows[:, None] * R + offs_c[None, :]
    tl.store(hp, acc, mask=mrow[:, None])


@triton.jit
def _gemm_vllm(X, XQ, SX, PMAX, W, SW, HP, B_PTRS, SB_PTRS, BG, Y, TOK, SRT, SEG_IDS, SEG_N, SEG_START, N_SEG,
               M, N, K, R, S, END0, END1, END2, stride_xm, stride_xqm, stride_wn, stride_ym,
               X_FP8: tl.constexpr, SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, EVEN_K: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
               DOT_F32: tl.constexpr = False, RN: tl.constexpr = True, PRECISE: tl.constexpr = True,
               B_FP8: tl.constexpr = False, B_FP4: tl.constexpr = False, BASE: tl.constexpr = True,
               R_CHUNK: tl.constexpr = 64, A16: tl.constexpr = False):
    """BASE=True: y = FP8 GEMM + LoRA. BASE=False (hybrid): y += LoRA in place, y from the vendor GEMM; tiles without
    an adapter row are left untouched. A16=True (LOWBIT_A16_MAX_M): weight-only, W8A16 -- bf16 activations times the e4m3
    codes widened to bf16 in registers (exact), fp32 accumulation, the per-tensor scale once at the end: the matched
    engine's and the weight-matched trainer's arithmetic; no activation is rounded, so XQ, SX and PMAX are unused."""
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
    # When any token has an adapter, tiles run over vLLM's adapter-sorted token order (x_q rows gathered, y rows
    # scattered), so a tile meets a few contiguous adapter runs, not every adapter of an interleaved decode batch.
    # Without adapters, vLLM leaves the sorted indices stale: natural order then. Under CUDA graphs the batch is padded
    # to a captured size, but the metadata covers only the n_all real tokens: sorted positions [0, n_all) permute rows
    # [0, n_all), and positions past n_all hold stale indices (from earlier, larger batches: 10-09 illegal address).
    # Those positions run as their own rows, padding without adapter; indices are clamped as a last guard.
    n_seg = tl.load(N_SEG)
    n_ad = 0
    n_all = 0
    for i in range(0, n_seg):
        c = tl.load(SEG_N + i)
        n_all += c
        n_ad += tl.where(tl.load(SEG_IDS + i) >= 0, c, 0)
    srt_pos = mm & (n_ad > 0) & (offs_m < n_all)
    rows = tl.load(SRT + offs_m, mask=srt_pos, other=0)
    rows = tl.where(srt_pos, tl.minimum(tl.maximum(rows, 0), M - 1), offs_m)
    tok = tl.load(TOK + rows, mask=srt_pos, other=-1)
    k_end = K
    if BASE:   # (Triton 3.7 rejects `not` on a constexpr: plain if/else)
        pass
    else:
        k_end = 0
        if tl.max(tok, axis=0) < 0:
            return
    if A16:
        sx = tl.full((BLOCK_M,), 1.0, tl.float32)
    elif X_FP8:
        sx = tl.load(SX + rows, mask=mm, other=1.0)
    else:
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for j in tl.static_range(SPLIT_K1):
            amax = tl.maximum(amax, tl.load(PMAX + j * M + rows, mask=mm, other=0.0))
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            ma = mm[:, None]
            mb = mn[None, :]
        else:
            ma = mm[:, None] & (offs_k[None, :] < K)
            mb = mn[None, :] & (offs_k[:, None] < K)
        if A16:
            a = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=ma, other=0.0)
        elif X_FP8:
            a = tl.load(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_k[None, :], mask=ma, other=0.0)
        else:
            xb = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=ma, other=0.0).to(tl.float32)
            a = to_e4m3(tl.clamp(tl.math.div_rn(xb, sx[:, None]), -_FP8_MAX, _FP8_MAX), RN)
        b = tl.load(W + offs_n[None, :] * stride_wn + offs_k[:, None], mask=mb, other=0.0)
        if A16:
            if DOT_F32:
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
            else:
                acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16), acc)
        elif DOT_F32:
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
        elif PRECISE:   # FP8 MMA accumulates with reduced precision: promote to fp32 after every K-block, as CUTLASS does
            acc += tl.dot(a, b)
        else:
            acc = tl.dot(a, b, acc)
    if BASE:
        acc = acc * (sx[:, None] * tl.load(SW))
    else:
        acc = tl.load(Y + rows[:, None].to(tl.int64) * stride_ym + offs_n[None, :], mask=mm[:, None] & mn[None, :],
                      other=0.0).to(tl.float32)
    if tl.max(tok, axis=0) >= 0:                                       # some row of this tile has an adapter
        n0 = pid_n * BLOCK_N
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32) + (n0 >= END2).to(tl.int32)   # up to 4 slices
        n_s0 = tl.where(s == 0, 0, tl.where(s == 1, END0, tl.where(s == 2, END1, END2)))         # first column of the slice
        n_s = tl.where(s == 0, END0, tl.where(s == 1, END1 - END0, tl.where(s == 2, END2 - END1, N - END2)))   # slice width
        if B_FP8:   # FP8-stored adapter (E4M3, one fp32 scale per output row): exact upcast, scale after the dot
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.float8e4nv))
            sb_base = tl.load(SB_PTRS + s).to(tl.pointer_type(tl.float32))
        elif B_FP4:   # NVFP4-stored adapter ([N_s, R/2] codes along R, [N_s, R/16] block scales, a global)
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.uint8))
            bs_base = tl.load(SB_PTRS + s).to(tl.pointer_type(tl.float8e4nv))
        else:
            b_base = tl.load(B_PTRS + s).to(tl.pointer_type(tl.bfloat16))
        hm = mm & (tok >= 0)
        t_lo = pid_m * BLOCK_M
        # The rank in chunks of RC (R_CHUNK < BLOCK_R where the host asks for it: NVFP4-stored B and the expand-add,
        # whose whole-rank tiles spilled registers at rank 128, job 65486195): h, its bf16 copy and B's dequantized tile
        # are [., RC]. The split-K partials are summed one at a time (unrolled, all were live at once).
        RC: tl.constexpr = min(BLOCK_R, R_CHUNK)
        for rc in tl.static_range(0, BLOCK_R, RC):
            offs_r = rc + tl.arange(0, RC)
            mr = offs_r < R
            h = tl.zeros((BLOCK_M, RC), dtype=tl.float32)
            for j in range(0, SPLIT_K1):
                h += tl.load(HP + j * (S * M * R) + s * (M * R) + rows[:, None] * R + offs_r[None, :],
                             mask=hm[:, None] & mr[None, :], other=0.0)
            hb = h.to(tl.bfloat16)
            for i in range(0, n_seg):
                lora = tl.load(SEG_IDS + i)
                cnt = tl.load(SEG_N + i)
                st = tl.load(SEG_START + i)
                # sorted positions [st, st + cnt) belong to this segment: skip segments that miss this tile
                if (lora >= 0) & (cnt > 0) & (st < t_lo + BLOCK_M) & (st + cnt > t_lo):
                    sel = tok == lora
                    if tl.sum(sel.to(tl.int32), axis=0) > 0:
                        h_l = tl.where(sel[:, None], hb, 0.0).to(tl.bfloat16)
                        # Low-precision B: dequantize the tile with its scales folded in (one bf16 rounding, <= 2^-9
                        # relative, far below the storage error), so the expand still accumulates in place.
                        if B_FP4:
                            bg = tl.load(BG + s * (n_seg - 1) + lora)
                            if RC >= 64:   # vectorized loads + join/reshape (correct from 64-wide tiles up)
                                offs_rp = rc // 2 + tl.arange(0, RC // 2)
                                offs_rs = rc // 16 + tl.arange(0, RC // 16)
                                bp = tl.load(b_base + lora.to(tl.int64) * n_s * (R // 2)
                                             + (offs_n[:, None] - n_s0) * (R // 2) + offs_rp[None, :],
                                             mask=mn[:, None] & (offs_rp < R // 2)[None, :], other=0)
                                bsc = tl.load(bs_base + lora.to(tl.int64) * n_s * (R // 16)
                                              + (offs_n[:, None] - n_s0) * (R // 16) + offs_rs[None, :],
                                              mask=mn[:, None] & (offs_rs < R // 16)[None, :], other=0.0)
                                bt = tl.trans(_dequant_tile(bp, bsc, BLOCK_N, RC) * bg).to(tl.bfloat16)
                            else:   # narrow rank blocks: the join/reshape form miscompiles when fed to a dot (16-wide
                                # tiles, tests/gpu_dequant_check.py), so gather each element's byte, nibble and scale
                                byte = tl.load(b_base + lora.to(tl.int64) * n_s * (R // 2)
                                               + (offs_n[None, :] - n_s0) * (R // 2) + (offs_r // 2)[:, None],
                                               mask=mn[None, :] & mr[:, None], other=0)
                                nib = (byte.to(tl.int32) >> ((offs_r % 2) * 4)[:, None]) & 15
                                bsc = tl.load(bs_base + lora.to(tl.int64) * n_s * (R // 16)
                                              + (offs_n[None, :] - n_s0) * (R // 16) + (offs_r // 16)[:, None],
                                              mask=mn[None, :] & mr[:, None], other=0.0)
                                bt = (_e2m1_value(nib) * bsc.to(tl.float32) * bg).to(tl.bfloat16)   # [RC, BLOCK_N]
                        else:
                            bt = tl.load(b_base + lora.to(tl.int64) * n_s * R + (offs_n[None, :] - n_s0) * R
                                         + offs_r[:, None], mask=mn[None, :] & mr[:, None], other=0.0)
                            if B_FP8:
                                sbv = tl.load(sb_base + lora.to(tl.int64) * n_s + (offs_n - n_s0), mask=mn, other=0.0)
                                bt = (bt.to(tl.float32) * sbv[None, :]).to(tl.bfloat16)
                        if DOT_F32:
                            acc += tl.dot(h_l.to(tl.float32), bt.to(tl.float32))
                        else:
                            acc = tl.dot(h_l, bt, acc)
    tl.store(Y + rows[:, None].to(tl.int64) * stride_ym + offs_n[None, :], acc.to(tl.bfloat16),
             mask=mm[:, None] & mn[None, :])


def pick_cfg(M: int, N: int, K: int, R: int, widths, cc_major: int = 0) -> tuple[dict, dict]:
    """Shrink and GEMM configs by batch size, from the per-shape tuning of the Phase-0 kernel benchmark (L40S, Qwen3-8B
    and Llama-3.1-8B projections; results/p0_kernel.jsonl): small M wants 16-row tiles and long K-steps, large M
    128x256 tiles. BLOCK_N must divide every slice width (q/k/v, gate/up)."""
    if cc_major >= 10 and 16 < M <= 256:   # Blackwell (sm_120): GEMM tile search, job 65488328
        cg = {32: {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
              64: {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3},
              128: {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 8, "num_stages": 3},
              256: {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}}[
            next(b for b in (32, 64, 128, 256) if M <= b)].copy()
    elif M <= 32:
        bn = 32 if (K >= 8192 or N >= 16384) else 64
        cg = {"BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": 256, "num_warps": 4, "num_stages": 4 if bn == 32 else 3}
    elif M <= 64:
        cg = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    elif M <= 128:
        cg = {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4}
    elif M <= 256:
        cg = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4}
    else:
        cg = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3}
    while any(w % cg["BLOCK_N"] for w in widths) and cg["BLOCK_N"] > 32:
        cg["BLOCK_N"] //= 2
    cg["GROUP_M"] = 8
    cs = {"BLOCK_M": 16 if M <= 1024 else 64, "BLOCK_SR": min(64, R), "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    return cs, cg


class Scratch:
    """Scratch shared by every fused layer of a model: layers run one after another on one stream (also inside a
    CUDA graph), so one buffer of each kind, sized for the largest layer, serves all of them. Sizes: h partials
    max(split_k * S * 64, S * max_m) * R (split-K only in decode mode, M <= 64), maxima split_k * max_m, x_q max_m * K."""

    def __init__(self, device):
        self.device = device
        self.hp = torch.empty(0, dtype=torch.float32, device=device)
        self.pmax = torch.empty(0, dtype=torch.float32, device=device)
        self.xq = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)
        self.sx = torch.empty(0, dtype=torch.float32, device=device)

    def ensure(self, split_k: int, S: int, R: int, max_m: int, K: int) -> None:
        need_hp = max(split_k * S * min(SPLIT_M_MAX, max_m), S * max_m) * R
        if self.hp.numel() < need_hp:
            self.hp = torch.empty(need_hp, dtype=torch.float32, device=self.device)
        if self.pmax.numel() < split_k * max_m:
            self.pmax = torch.empty(split_k * max_m, dtype=torch.float32, device=self.device)
        if self.xq.numel() < max_m * K:
            self.xq = torch.empty(max_m * K, dtype=torch.float8_e4m3fn, device=self.device)
        if self.sx.numel() < max_m:
            self.sx = torch.empty(max_m, dtype=torch.float32, device=self.device)


_SCRATCH: dict = {}


def shared_scratch(device) -> Scratch:
    key = str(device)
    if key not in _SCRATCH:
        _SCRATCH[key] = Scratch(device)
    return _SCRATCH[key]


_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 1 << 30)
_DISPATCH: dict = {}   # (gpu, N, K, R, widths, adapter format, hybridq?) -> {bucket: path}, shared by same-shape layers


def _bucket(M: int) -> int:
    return next(b for b in _BUCKETS if M <= b)


def _graph_us(fn, reps: int = 10) -> float:
    """Median microseconds of one CUDA-graph replay of fn (as vLLM runs a decode step)."""
    import statistics
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(1000 * e0.elapsed_time(e1))
    del g
    return statistics.median(ts)


# HOST-OVERHEAD (2026-10-25; opt-in LOWBIT_FAST_LAUNCH=1, every earlier run unchanged): in vLLM's steps above its CUDA-graph
# capture bound each call of the op runs eagerly, and Triton's per-launch Python (argument binding and specialization, the cache-key
# hash, the global-value check) is most of its host time. The fast path launches the compiled kernel a call's specialization
# compiled to directly; _run_fast keys it on everything that varies per call and that Triton specializes on.
_FAST_LAUNCH = os.environ.get("LOWBIT_FAST_LAUNCH", "0") not in ("0", "", "false", "False")


def _launch(cache: dict, key, kern, grid, args: tuple, kw: dict) -> None:
    """kern[grid](*args, **kw); after the first call for key, the cached compiled kernel launched directly (all parameters in
    declaration order, constexprs included, as Triton's own launch passes them; no launch hooks)."""
    ent = cache.get(key)
    if ent is None:
        ck = kern[grid](*args, **kw)
        names = kern.arg_names
        tail = []
        for n, prm in zip(names[len(args):], kern.params[len(args):]):
            if n in kw:
                tail.append(kw[n])
            elif prm.default is not prm._param.empty:
                tail.append(prm.default)
            else:                          # not bindable positionally: keep Triton's path for this key
                cache[key] = False
                return
        cache[key] = (ck, tuple(tail))
        return
    if ent is False:
        kern[grid](*args, **kw)
        return
    ck, tail = ent
    g1 = grid[1] if len(grid) > 1 else 1
    g2 = grid[2] if len(grid) > 2 else 1
    ck.run(grid[0], g1, g2, torch._C._cuda_getCurrentRawStream(args[0].device.index), ck.function, ck.packed_metadata, None,
           None, None, *args, *tail)


class VllmFusedFP8LoRA:
    """Fused base + LoRA for one vLLM LoRA layer whose base is online per-tensor FP8.

    w_q [N, K] float8_e4m3fn (= base_layer.weight.t()), s_w [1]; a_stacked / b_stacked: vLLM's tuples of per-slice
    buffers [L, 1, R, K] / [L, 1, N_s, R] (bf16); max_m = the engine's max tokens per forward (static scratch).
    lora_fp8: serve FP8-stored adapters (E4M3 with one fp32 scale per row of A and of B, as P0-NUMERICS's lora_fp8),
    kept in our own stacks; requantize(index) refreshes them from vLLM's bf16 slots after vLLM writes a slot.
    """

    def __init__(self, w_q, s_w, a_stacked, b_stacked, max_m, *, split_k=None, lora_fp8=False, lora_fp4=False):
        self.w_q, self.s_w = w_q, s_w
        self.N, self.K = w_q.shape
        self.S = len(a_stacked)
        self.L, _, self.R, k_a = a_stacked[0].shape
        assert k_a == self.K and all(t.dtype == torch.bfloat16 for t in (*a_stacked, *b_stacked))
        widths = [t.shape[2] for t in b_stacked]
        assert sum(widths) == self.N and self.S <= 4
        ends = [sum(widths[:i + 1]) for i in range(self.S)]
        self.end0 = ends[0] if self.S >= 2 else self.N
        self.end1 = ends[1] if self.S >= 3 else self.N
        self.end2 = ends[2] if self.S >= 4 else self.N             # a 4th slice: e.g. a linear-attention in_proj
        self.widths = widths
        dev = w_q.device
        self.a_stacked, self.b_stacked = a_stacked, b_stacked       # keep the buffers alive
        self.a_ptrs = torch.tensor([t.data_ptr() for t in a_stacked], dtype=torch.int64, device=dev)
        self.b_ptrs = torch.tensor([t.data_ptr() for t in b_stacked], dtype=torch.int64, device=dev)
        self.stride_al, self.stride_am = a_stacked[0].stride(0), a_stacked[0].stride(2)
        self.split_k = split_k or max(1, min(16, self.K // 512))
        self.cc_major = torch.cuda.get_device_capability(dev)[0] if dev.type == "cuda" else 0
        # Hopper: the TMA-fed GEMM with the LoRA epilogue (fp8_lora_tma.py), opt-in with LOWBIT_TMA=1 (MLSys v2)
        self.use_tma = self.cc_major == 9 and os.environ.get("LOWBIT_TMA", "0") == "1"
        self.tma_cfg = None
        self.max_m = max_m
        self.scratch = shared_scratch(dev)
        self.scratch.ensure(self.split_k, self.S, self.R, max_m, self.K)
        self.n_seg = torch.full((1,), self.L + 1, dtype=torch.int32, device=dev)
        self.dot_f32 = False
        self.r_chunk = None        # the GEMM epilogue's rank chunk: None = auto (_run)
        self.base_fn = None        # the vendor FP8 GEMM (x -> y), set by the plugin: enables the hybrid path
        self.base_q_fn = None      # the vendor GEMM on our codes ((x_q [M, K] e4m3, s_x [M, 1]) -> y): "hybridq",
        # the hybrid path without vLLM's quantize kernel (K1's all-token pass writes the same codes and scales)
        self.dispatch = {}         # bucket of M -> "fused" | "hybrid" | "hybridq" | "hybrids", from tune_dispatch()
        # LOWBIT_A16_MAX_M (default 0: off): batches of at most this many tokens run the fused path weight-only (W8A16, the
        # matched engine's arithmetic: bf16 activations, no quantization pass) instead of the dispatch's choice
        self.a16_max_m = int(os.environ.get("LOWBIT_A16_MAX_M", "0") or 0)
        # "hybrids": the vendor GEMM, then vLLM's own LoRA kernels on vLLM's bf16 adapters, so the timed dispatch never
        # picks something slower than the stock path (bf16 adapters only; CPU flags as LoRAKernelMeta's defaults)
        self._no_lora_cpu = torch.zeros(1, dtype=torch.bool)
        self._n_active_cpu = torch.tensor([self.L + 1], dtype=torch.int32)
        self.lora_fp8, self.lora_fp4 = lora_fp8, lora_fp4
        self.staged = False        # release(): a_stacked / b_stacked are one staging slot, not vLLM's L slots
        assert not (lora_fp8 and lora_fp4)
        self.ag = self.bg = torch.zeros(1, dtype=torch.float32, device=dev)   # unused unless lora_fp4
        if lora_fp4:   # NVFP4-stored adapters: codes and block scales along K (A) and along R (B)
            assert self.R % 16 == 0 and self.K % 16 == 0
            self.a4 = tuple(torch.zeros(self.L, 1, self.R, self.K // 2, dtype=torch.uint8, device=dev) for _ in a_stacked)
            self.a4s = tuple(torch.zeros(self.L, 1, self.R, self.K // 16, dtype=torch.float8_e4m3fn, device=dev)
                             for _ in a_stacked)
            self.b4 = tuple(torch.zeros(self.L, 1, w, self.R // 2, dtype=torch.uint8, device=dev) for w in widths)
            self.b4s = tuple(torch.zeros(self.L, 1, w, self.R // 16, dtype=torch.float8_e4m3fn, device=dev)
                             for w in widths)
            self.ag = torch.zeros(self.S, self.L, dtype=torch.float32, device=dev)
            self.bg = torch.zeros(self.S, self.L, dtype=torch.float32, device=dev)
            self.a_ptrs = torch.tensor([t.data_ptr() for t in self.a4], dtype=torch.int64, device=dev)
            self.b_ptrs = torch.tensor([t.data_ptr() for t in self.b4], dtype=torch.int64, device=dev)
            self.sa_ptrs = torch.tensor([t.data_ptr() for t in self.a4s], dtype=torch.int64, device=dev)
            self.sb_ptrs = torch.tensor([t.data_ptr() for t in self.b4s], dtype=torch.int64, device=dev)
            self.requantize()
        elif lora_fp8:
            self.a8 = tuple(torch.zeros(t.shape, dtype=torch.float8_e4m3fn, device=dev) for t in a_stacked)
            self.b8 = tuple(torch.zeros(t.shape, dtype=torch.float8_e4m3fn, device=dev) for t in b_stacked)
            self.sa = tuple(torch.zeros(self.L, self.R, dtype=torch.float32, device=dev) for _ in a_stacked)
            self.sb = tuple(torch.zeros(self.L, t.shape[2], dtype=torch.float32, device=dev) for t in b_stacked)
            assert all(t.stride() == u.stride() for t, u in zip(self.a8, a_stacked))
            self.a_ptrs = torch.tensor([t.data_ptr() for t in self.a8], dtype=torch.int64, device=dev)
            self.b_ptrs = torch.tensor([t.data_ptr() for t in self.b8], dtype=torch.int64, device=dev)
            self.sa_ptrs = torch.tensor([t.data_ptr() for t in self.sa], dtype=torch.int64, device=dev)
            self.sb_ptrs = torch.tensor([t.data_ptr() for t in self.sb], dtype=torch.int64, device=dev)
            self.requantize()
        else:
            self.sa_ptrs = self.sb_ptrs = self.a_ptrs                  # unused (A_FP8 / B_FP8 off)

    def release(self, a_stage, b_stage) -> None:
        """Serve from the low-precision copies alone (as AdapterStore.release): vLLM writes every adapter through one
        staging slot (a_stage[s] [1, 1, R, K], b_stage[s] [1, 1, N_s, R]), which requantize(index) reads."""
        assert self.lora_fp8 or self.lora_fp4
        self.a_stacked, self.b_stacked, self.staged = a_stage, b_stage, True

    @torch.no_grad()
    def requantize(self, index: int | None = None) -> None:
        """FP8 copies of vLLM's bf16 slots (all slots, or one): per-row scale amax / 448, E4M3 round to nearest."""
        if not (self.lora_fp8 or self.lora_fp4):
            return
        assert not (self.staged and index is None)
        idx = slice(None) if index is None else slice(index, index + 1)
        sidx = slice(0, 1) if self.staged else idx                    # the source: the staging slot after release()
        if self.lora_fp4:
            for s in range(self.S):
                for src, q, qs, g in ((self.a_stacked, self.a4, self.a4s, self.ag),
                                      (self.b_stacked, self.b4, self.b4s, self.bg)):
                    packed, sf, inv_g = nvfp4_quant_batched(src[s][sidx, 0])
                    q[s][idx, 0], qs[s][idx, 0], g[s, idx] = packed, sf, inv_g
            return
        for src, q, sc in ((self.a_stacked, self.a8, self.sa), (self.b_stacked, self.b8, self.sb)):
            for s in range(self.S):
                t = src[s][sidx, 0].float()                             # [l, rows, cols]
                scale = (t.abs().amax(-1) / FP8_MAX).clamp_min(1e-12)
                q[s][idx, 0] = (t / scale[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                sc[s][idx] = scale

    def _dummy_meta(self, M: int, lora: int):
        dev = self.w_q.device
        tok = torch.full((M,), lora, dtype=torch.int32, device=dev)
        srt = torch.arange(M, dtype=torch.int32, device=dev)
        seg_ids = torch.full((self.L + 1,), -1, dtype=torch.int32, device=dev)
        seg_n = torch.zeros(self.L + 1, dtype=torch.int32, device=dev)
        seg_start = torch.zeros(self.L + 2, dtype=torch.int32, device=dev)   # L + 2 as LoRAKernelMeta (vLLM's kernels)
        if lora >= 0:
            seg_ids[0], seg_n[0] = lora, M
            seg_start[1] = M
        return tok, srt, seg_n, seg_start, seg_ids

    def _dummy_meta_multi(self, M: int, k: int):
        """k distinct adapters (slots 0..k-1), tokens round-robin and sorted by adapter, as LoRAKernelMeta lays them
        out (token map, sorted token indices, tokens per adapter, start offsets, active ids)."""
        dev = self.w_q.device
        k = max(1, min(k, self.L, M))
        tok = (torch.arange(M, device=dev) % k).to(torch.int32)
        srt = torch.argsort(tok, stable=True).to(torch.int32)
        counts = torch.bincount(tok.long(), minlength=k).to(torch.int32)
        seg_ids = torch.full((self.L + 1,), -1, dtype=torch.int32, device=dev)
        seg_n = torch.zeros(self.L + 1, dtype=torch.int32, device=dev)
        seg_start = torch.zeros(self.L + 2, dtype=torch.int32, device=dev)
        seg_ids[:k] = torch.arange(k, dtype=torch.int32, device=dev)
        seg_n[:k] = counts
        seg_start[1:k + 1] = torch.cumsum(counts, 0).to(torch.int32)
        return tok, srt, seg_n, seg_start, seg_ids

    def tune_dispatch(self) -> dict:
        """Per bucket of M, time the fused path (our GEMM with the LoRA epilogue) against the hybrid (the vendor GEMM,
        then our shrink and expand-add kernels), both CUDA-graph replayed, and keep the faster. Our Triton FP8 GEMM beats
        vLLM's quantize + CUTLASS on Ada at every M but trails CUTLASS on Hopper, especially at large M (job 65486657).
        Same-shape layers share one table (timed once per shape)."""
        if self.base_fn is None:
            return self.dispatch
        # LOWBIT_DISPATCH_K=k (experimental): time each path with one adapter and with k distinct adapters, and keep
        # the path with the smaller sum (the default, k = 1, times one adapter only)
        kd = int(os.environ.get("LOWBIT_DISPATCH_K", "1"))
        key = (torch.cuda.get_device_name(self.w_q.device), self.N, self.K, self.R, tuple(self.widths),
               "fp4" if self.lora_fp4 else "fp8" if self.lora_fp8 else "bf16", self.base_q_fn is not None, kd)
        # LOWBIT_TUNE_MAX_M (default 2048): buckets above it are not timed and inherit the last timed bucket's path. The
        # timing runs inside vLLM's memory-profiling forward (the op is built lazily), so its large-M temporaries were
        # counted as activation memory: on an L40S with Qwen3.6-27B, timing up to the engine's max tokens cost 1.75 GiB of
        # KV cache (diagnostic 10-14). Ada's fused path wins at every timed M and Hopper's hybrid at large M, so the
        # largest timed bucket's choice is the one to extend.
        tune_max = int(os.environ.get("LOWBIT_TUNE_MAX_M", "2048"))
        if key not in _DISPATCH:
            table = {}
            for b in _BUCKETS:
                M = min(b, self.max_m)
                if M < 1 or (table and M == prev):
                    break
                if table and M > tune_max:                      # untimed: inherit the largest timed bucket's path
                    table[b] = table[max(table)]
                    prev = M
                    continue
                prev = M
                x = torch.randn(M, self.K, device=self.w_q.device).to(torch.bfloat16)
                metas = [self._dummy_meta(M, 0)] + ([self._dummy_meta_multi(M, kd)] if min(kd, self.L, M) > 1 else [])
                tm = lambda h: sum(_graph_us(lambda m=m: self._run(x, m, hybrid=h)) for m in metas)   # noqa: E731
                t = {"fused": tm(False), "hybrid": tm(True)}
                if self.base_q_fn is not None:
                    t["hybridq"] = tm("q")
                if not (self.lora_fp8 or self.lora_fp4):
                    t["hybrids"] = tm("s")
                table[b] = min(t, key=t.get)
                del x, metas
            _DISPATCH[key] = table
            torch.cuda.empty_cache()                            # return the timing's segments before KV sizing
        self.dispatch = _DISPATCH[key]
        return self.dispatch

    def warmup(self) -> None:
        """Compile every kernel variant the config table can select (Triton compiles lazily, and a first compile inside
        a serving step stalls it for seconds). Dummy inputs: all tokens on adapter slot 0, plus a no-adapter batch.
        With a vendor GEMM set, both paths are compiled and the dispatch table is timed first."""
        self.tune_dispatch()
        dev = self.w_q.device
        # Triton specializes integer arguments on ==1 and on divisibility by 16: one divisible and one non-divisible
        # batch size per config bucket of pick_cfg (<=32, <=64, <=128, <=256, <=1024, >1024), plus M = 1
        sizes = [m for m in (1, 2, 16, 17, 48, 49, 96, 97, 192, 193, 512, 513, 2048, 2049, self.max_m, self.max_m - 1)
                 if 1 <= m <= self.max_m]
        for M in sizes:
            x = torch.zeros(M, self.K, dtype=torch.bfloat16, device=dev)
            for lora in (0, -1):
                tok, srt, seg_n, seg_start, seg_ids = self._dummy_meta(M, lora)   # (L + 2 start offsets, as vLLM's)
                self(x, (tok, srt, seg_n, seg_start, seg_ids))
                if self.base_fn is not None:   # compile the other paths too
                    for h in ((False, True) + (("q",) if self.base_q_fn is not None else ())
                              + (("s",) if not (self.lora_fp8 or self.lora_fp4) else ())):
                        self._run(x, (tok, srt, seg_n, seg_start, seg_ids), hybrid=h)
        torch.cuda.synchronize()

    def __call__(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None):
        """meta_args: LoRAKernelMeta.meta_args(M, ...) = (token_lora_mapping, token_indices_sorted_by_lora_ids,
        num_tokens_per_lora, lora_token_start_loc, active_lora_ids, no_lora_flag_cpu, num_active_loras).
        The path (fused, hybrid or hybridq) follows the dispatch table for M's bucket; fused when there is no table;
        weight-only fused (W8A16) for M <= a16_max_m."""
        if self.a16_max_m and x.shape[0] <= self.a16_max_m:
            return self._run(x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g, a16=True)
        d = self.dispatch.get(_bucket(x.shape[0])) if self.base_fn is not None else None
        hybrid = ("q" if (d == "hybridq" and self.base_q_fn is not None) else "s" if d == "hybrids"
                  else d == "hybrid")
        return self._run(x, meta_args, out=out, mode=mode, cfg_s=cfg_s, cfg_g=cfg_g, hybrid=hybrid)

    def _run_stock(self, x, meta_args):
        """The vendor GEMM, then vLLM's own LoRA shrink and expand (Punica kernels) on vLLM's bf16 adapter buffers."""
        from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
        tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
        M = x.shape[0]
        y = self.base_fn(x)
        hb = self.scratch.hp[: self.S * M * self.R].view(self.S, M, self.R)
        hb.zero_()
        m7 = (tok, srt, seg_n, seg_start, seg_ids, self._no_lora_cpu, self._n_active_cpu)
        lora_shrink(x, list(self.a_stacked), hb, *m7, 1.0)
        lora_expand(hb, list(self.b_stacked), y, *m7, offset_start=0, add_inputs=True)
        return y

    def _fast_plan(self, M: int, hybrid, a16: bool) -> tuple:
        """_run's per-M decisions (tiles, split, grids, constexprs), computed once per (M, path)."""
        sc = self.scratch
        cs_t, split = shrink_choice(M, self.R, self.split_k, self.cc_major)
        while split > 1 and (split * self.S * M * self.R > sc.hp.numel() or split * M > sc.pmax.numel()):
            split //= 2
        prefill = M > 64
        ds, cg = pick_cfg(M, self.N, self.K, self.R, self.widths, self.cc_major)
        cs = cs_t or ds
        assert self.R % cs["BLOCK_SR"] == 0 and all(w % cg["BLOCK_N"] == 0 for w in self.widths)
        n_sr = self.S * self.R // cs["BLOCK_SR"]
        grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, self.L + (1 if (hybrid is True or a16) else 2))
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        br = max(16, triton.next_power_of_2(self.R))
        rck = self.r_chunk or (64 if (self.lora_fp4 or hybrid) else br)
        precise = os.environ.get("LOWBIT_PRECISE", "1") != "0"
        kw_s = dict(SPLIT_K=split, WRITE_XQ=prefill, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
                    BLOCK_K=cs["BLOCK_K"], DOT_F32=self.dot_f32, RN=not self.dot_f32, A_FP8=self.lora_fp8, A_FP4=self.lora_fp4,
                    num_warps=cs["num_warps"], num_stages=cs["num_stages"])
        kw_g = dict(X_FP8=not a16, A16=a16, SPLIT_K1=split, BLOCK_R=br, EVEN_K=self.K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"],
                    BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"], PRECISE=precise, DOT_F32=self.dot_f32,
                    RN=not self.dot_f32, B_FP8=self.lora_fp8, B_FP4=self.lora_fp4, BASE=not hybrid, R_CHUNK=rck,
                    num_warps=cg["num_warps"], num_stages=cg["num_stages"])
        return split, grid_s, grid_g, kw_s, kw_g

    def _run_fast(self, x, meta_args, out, hybrid, a16):
        """_run (mode and tiles by M, no TMA) with LOWBIT_FAST_LAUNCH=1: the per-M decisions cached (_fast_plan) and both
        kernels launched through _launch, keyed on what varies per call and Triton specializes on: M == 1 and M % 16 (the plan
        is per M), and the 16-byte alignment and leading strides of x, y and vLLM's metadata tensors; every other argument is
        fixed per op (its weights, adapters and shared scratch, whose buffers only ever move to other aligned allocations)."""
        tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
        M = x.shape[0]
        plans = self.__dict__.setdefault("_fast_plans", {})
        cks = self.__dict__.setdefault("_fast_cks", {})
        pk = (M, hybrid, a16)
        plan = plans.get(pk)
        if plan is None:
            plan = plans[pk] = self._fast_plan(M, hybrid, a16)
        split, grid_s, grid_g, kw_s, kw_g = plan
        sc = self.scratch
        hp, pmax, sxb = sc.hp, sc.pmax, sc.sx
        xq = sc.xq[: M * self.K].view(M, self.K)
        al = (x.data_ptr() % 16 == 0, x.stride(0) % 16 == 0, tok.data_ptr() % 16 == 0, srt.data_ptr() % 16 == 0,
              seg_ids.data_ptr() % 16 == 0, seg_n.data_ptr() % 16 == 0, seg_start.data_ptr() % 16 == 0)
        _launch(cks, ("s", pk) + al, _shrink_vllm, grid_s,
                (x, self.a_ptrs, self.sa_ptrs, self.ag, hp, pmax, xq, sxb, srt, seg_ids, seg_n, seg_start,
                 M, self.K, self.R, self.S, self.L + 1, x.stride(0), self.stride_al, self.stride_am, xq.stride(0)), kw_s)
        if hybrid == "q":
            y = self.base_q_fn(xq, sxb[:M].view(M, 1))
        elif hybrid:
            y = self.base_fn(x)
        else:
            y = out if out is not None else torch.empty(M, self.N, dtype=torch.bfloat16, device=x.device)
        _launch(cks, ("g", pk, y.data_ptr() % 16 == 0, y.stride(0) % 16 == 0) + al, _gemm_vllm, grid_g,
                (x, xq, sxb, pmax, self.w_q, self.s_w, hp, self.b_ptrs, self.sb_ptrs, self.bg, y, tok, srt, seg_ids, seg_n,
                 seg_start, self.n_seg, M, self.N, self.K, self.R, self.S, self.end0, self.end1, self.end2,
                 x.stride(0), xq.stride(0), self.w_q.stride(0), y.stride(0)), kw_g)
        return y

    def _run(self, x, meta_args, out=None, mode=None, cfg_s=None, cfg_g=None, hybrid=False, split=None, a16=False):
        if hybrid == "s":
            return self._run_stock(x, meta_args)
        if _FAST_LAUNCH and split is None and mode is None and cfg_s is None and cfg_g is None and not self.use_tma:
            return self._run_fast(x, meta_args, out, hybrid, a16)
        tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
        M = x.shape[0]
        sc = self.scratch
        hp, pmax, sxb = sc.hp, sc.pmax, sc.sx
        xq = sc.xq[: M * self.K].view(M, self.K)
        if split is not None:   # explicit override (tile search): any split the shared scratch holds
            assert split == 1 or (split * self.S * M * self.R <= hp.numel() and split * M <= pmax.numel())
            mode = mode or ("decode" if M <= 64 else "prefill")
        elif mode is None:   # tuned shrink tiles and split per architecture (common.shrink_choice)
            cs_t, split = shrink_choice(M, self.R, self.split_k, self.cc_major)
            while split > 1 and (split * self.S * M * self.R > hp.numel() or split * M > pmax.numel()):
                split //= 2                                              # what the shared scratch holds
            cfg_s = cfg_s or cs_t
            mode = "decode" if M <= 64 else "prefill"
        else:
            split = 1 if mode == "prefill" else self.split_k
        prefill = mode == "prefill"
        ds, dg = pick_cfg(M, self.N, self.K, self.R, self.widths, self.cc_major)
        cs = cfg_s or ds
        cg = cfg_g or dg
        assert self.R % cs["BLOCK_SR"] == 0 and all(w % cg["BLOCK_N"] == 0 for w in self.widths)
        n_sr = self.S * self.R // cs["BLOCK_SR"]
        # axis 2: the L + 1 adapter segments, then L1 = the all-token x_q pass (skipped by the plain hybrid path, whose
        # vendor GEMM quantizes x itself; hybrid="q" feeds our codes to the vendor GEMM instead)
        grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, self.L + (1 if (hybrid is True or a16) else 2))
        _shrink_vllm[grid_s](
            x, self.a_ptrs, self.sa_ptrs, self.ag, hp, pmax, xq, sxb, srt, seg_ids, seg_n, seg_start,
            M, self.K, self.R, self.S, self.L + 1, x.stride(0), self.stride_al, self.stride_am, xq.stride(0),
            SPLIT_K=split, WRITE_XQ=prefill, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
            BLOCK_K=cs["BLOCK_K"], DOT_F32=self.dot_f32, RN=not self.dot_f32, A_FP8=self.lora_fp8, A_FP4=self.lora_fp4,
            num_warps=cs["num_warps"], num_stages=cs["num_stages"])
        if hybrid == "q":
            y = self.base_q_fn(xq, sxb[:M].view(M, 1))
        elif hybrid:
            y = self.base_fn(x)
        else:
            y = out if out is not None else torch.empty(M, self.N, dtype=torch.bfloat16, device=x.device)
        if self.use_tma and not hybrid and not a16:
            from lowbit_lora.kernels.fp8_lora_tma import gemm_tma
            return gemm_tma(self, x, xq, sxb, hp, y, tok, seg_ids, seg_n, split, cfg=self.tma_cfg,
                            precise=os.environ.get("LOWBIT_PRECISE", "1") != "0")
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        br = max(16, triton.next_power_of_2(self.R))
        # The epilogue's rank chunk: 64 for NVFP4-stored B and for the expand-add (BASE=False), whose whole-rank tiles
        # spill at rank 128 (L40S, rank 128, 1024-4096 tokens: 1.2-1.4x faster); the whole rank otherwise, where one
        # dot is faster (bf16/FP8 fused: 1.15-1.25x slower in chunks). r_chunk forces a chunk (tests).
        rck = self.r_chunk or (64 if (self.lora_fp4 or hybrid) else br)
        _gemm_vllm[grid_g](
            x, xq, sxb, pmax, self.w_q, self.s_w, hp, self.b_ptrs, self.sb_ptrs, self.bg, y, tok, srt, seg_ids, seg_n,
            seg_start,
            self.n_seg, M, self.N, self.K, self.R, self.S, self.end0, self.end1, self.end2,
            x.stride(0), xq.stride(0), self.w_q.stride(0), y.stride(0),
            X_FP8=not a16, A16=a16, SPLIT_K1=split, BLOCK_R=br,
            EVEN_K=self.K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"],
            GROUP_M=cg["GROUP_M"], PRECISE=os.environ.get("LOWBIT_PRECISE", "1") != "0", DOT_F32=self.dot_f32,
            RN=not self.dot_f32, B_FP8=self.lora_fp8, B_FP4=self.lora_fp4, BASE=not hybrid, R_CHUNK=rck,
            num_warps=cg["num_warps"], num_stages=cg["num_stages"])
        return y
