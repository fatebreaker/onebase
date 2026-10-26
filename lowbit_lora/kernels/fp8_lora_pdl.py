"""Hopper decode path that takes the LoRA adapters off the critical path: a shrink kernel overlapped, through programmatic
dependent launch (PDL), with a swap-AB FP8 GEMM that quantizes its own activations and adds every adapter's expansion in its
epilogue.

Why. The plugin's fused path runs K1 (quantize x, every adapter's shrink h = x A^T) and then K2 (the FP8 GEMM, h_l B_l^T added
in its epilogue). On the H100 that chain is latency-bound at decode (HOPPER-SWAP-K2, results/swap_k2.jsonl: one rank-64
adapter adds 14-18 us per projection for 1-2 MB of adapter bytes): K2 cannot start before K1 has finished, and K2's epilogue
then waits on dependent cold loads (vLLM's segment metadata one segment at a time, the h partials, B). Here:

  shrink   K1's adapter segments (fp8_lora_vllm._shrink_vllm without the all-token segment): split-K fp32 partials of
           x A_l^T over vLLM's sorted segments. Launched first; every program executes griddepcontrol.launch_dependents at
           its start, so the GEMM launches at once. With X_MODE 1 or 2 its grid has one more segment of per-token
           quantization programs (one token per program, vLLM's per-token kernel structure, K1's arithmetic).
  GEMM     the swap-AB FP8 GEMM of fp8_lora_swap.py (weight rows on wgmma's 64-row side, tokens on its narrow side),
           launched with launch_pdl=True. It needs nothing from the shrink before its epilogue:
             X_MODE 0  it quantizes its own activations: a prologue takes each token's amax over the full row, the
                       mainloop quantizes each K-tile of x on the fly (K1's arithmetic: amax / 448 floored at
                       1 / (448 * 512), IEEE division, clamp to +-448, one saturating round-to-nearest E4M3 conversion);
             X_MODE 1  it reads x_q and the scales written by the shrink kernel's quantization programs, after an
                       acquire on a counter those programs release (no wait on the adapter shrinks);
             X_MODE 2  it waits for the whole shrink kernel (griddepcontrol.wait) before its mainloop (a baseline).
           Before its mainloop it reads vLLM's segment metadata in one vector load, finds the adapters present in its
           token tile, loads the first adapter's B tile (dequantized, scales folded in) into registers, and prefetches
           the other adapters' B tiles and its own first weight tiles into L2. It streams W while the shrink runs and
           executes griddepcontrol.wait only right before its epilogue reads h.
  split-K  where a projection has too few N-tiles (o, down): each CTA stores its fp32 partial and increments a per-tile
           arrival counter (release / acquire); the last CTA to arrive sums the partials in split order (fixed:
           deterministic, whichever CTA arrives last), applies the scales once, adds the LoRA term, stores bf16 and
           resets the counter for the next launch. No second kernel and no atomics on values.
  coop     the fallback if PDL does not overlap or cannot be captured: one kernel whose first programs run the shrink
           (and, with X_MODE 1, the quantization) and publish with release atomics, the rest run the GEMM and acquire
           the shrink counter before their epilogues (launched cooperatively where the grid fits, so every program is
           co-resident; the shrink programs also come first in launch order).

Arithmetic is the fused path's: e4m3 products accumulated in fp32 and promoted after every K-block (PRECISE, as CUTLASS),
the scales applied once, h summed over the shrink's split-K partials in split order and rounded to bf16 once, the
expansion as bf16 products accumulated in fp32, one rounding to bf16 at the end.

CPU interpreter (tests/pdl_k2_interp_check.py): GPU=False drops the griddepcontrol and prefetch instructions (inline PTX),
QRN=2 emulates the round-to-nearest E4M3 conversion exactly, DOT_F32=True uses fp32 dots. Hopper only: wgmma FP8 and PDL
need sm_90 (tests/pdl_k2_interp_check.py --compile checks the PTX).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.common import to_e4m3
from lowbit_lora.kernels.fp8_lora_vllm import _FP8_MAX, _MIN_SCALE
from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile, _e2m1_value

_BIG = tl.constexpr(2147483647)   # "no adapter" in min-reductions over adapter ids


# ----------------------------------------------------------------------------------------------------------- helpers
@triton.jit
def _e4m3_emul(v):
    """CPU interpreter: fp32 in [-448, 448] -> E4M3 with one round-to-nearest-even step, exactly (the interpreter's own
    fp8 cast truncates). The step of v's binade (2^-9 below the normal range); v / step rounded to an integer by adding and
    subtracting 2^23 (ties to even); the result is representable, so the final cast is exact."""
    a = tl.abs(v)
    e = ((a.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
    step = ((tl.maximum(e, -6) - 3 + 127) << 23).to(tl.float32, bitcast=True)
    r = ((a / step) + 8388608.0) - 8388608.0
    q = r * step
    q = tl.where(v.to(tl.int32, bitcast=True) < 0, -q, q)
    return q.to(tl.float8e4nv)


@triton.jit
def _quant_e4m3(x, sx, QRN: tl.constexpr):
    """K1's activation quantization (fp8_lora_vllm._shrink_vllm, all-token segment): x / s_x by IEEE division, clamped to
    +-448, one round-to-nearest-even E4M3 conversion. QRN 1: the PTX conversion (common.to_e4m3; GPU); 2: its exact
    emulation (CPU interpreter); 0: Triton's default conversion."""
    v = tl.clamp(tl.math.div_rn(x, sx), -_FP8_MAX, _FP8_MAX)
    if QRN == 1:
        q = to_e4m3(v, True)
    elif QRN == 2:
        q = _e4m3_emul(v)
    else:
        q = v.to(tl.float8e4nv)
    return q


@triton.jit
def _ptr16(TABLE, i, DT: tl.constexpr):
    """Pointer i of a device pointer table, as a pointer to DT, 16-byte aligned (torch allocations are): without the hint
    Triton cannot prove the alignment of a loaded pointer and emits scalar loads, unpipelined (tests: 32 two-byte loads
    per thread instead of 4 sixteen-byte ones for a 64 x 64 bf16 tile)."""
    return tl.multiple_of(tl.load(TABLE + i).to(tl.pointer_type(DT)), 16)


@triton.jit
def _prefetch_l2(ptrs):
    """prefetch.global.L2 for every address of ptrs: starts the DRAM read, holds no registers (GPU only)."""
    tl.inline_asm_elementwise("prefetch.global.L2 [$1]; // $0", "=r,l", [ptrs], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _b_tile(B_PTRS, SB_PTRS, BG, s, lora, ok, rc, offs_n, mn, n_s0, n_s, R, L,
            BLOCK_N: tl.constexpr, RC: tl.constexpr, B_FP8: tl.constexpr, B_FP4: tl.constexpr):
    """Adapter `lora`'s B tile for output rows offs_n (slice s) and rank columns [rc, rc + RC): bf16 [BLOCK_N, RC], storage
    scales folded in (one bf16 rounding, as every K2); zeros where not ok. Layouts as fp8_lora_vllm.VllmFusedFP8LoRA."""
    offs_r = rc + tl.arange(0, RC)
    mr = offs_r < R
    row = (offs_n - n_s0).to(tl.int64)
    l64 = lora.to(tl.int64)
    if B_FP4:   # NVFP4-stored B: [N_s, R/2] codes along R, [N_s, R/16] E4M3 block scales, one global per adapter and slice
        b_base = _ptr16(B_PTRS, s, tl.uint8)
        bs_base = _ptr16(SB_PTRS, s, tl.float8e4nv)
        bg = tl.load(BG + s * L + lora, mask=ok, other=0.0)
        if RC >= 64:   # vectorized loads + join/reshape (correct from 64-wide tiles up, tests/gpu_dequant_check.py)
            offs_rp = rc // 2 + tl.arange(0, RC // 2)
            offs_rs = rc // 16 + tl.arange(0, RC // 16)
            bp = tl.load(b_base + l64 * n_s * (R // 2) + row[:, None] * (R // 2) + offs_rp[None, :],
                         mask=ok & mn[:, None] & (offs_rp < R // 2)[None, :], other=0)
            bsc = tl.load(bs_base + l64 * n_s * (R // 16) + row[:, None] * (R // 16) + offs_rs[None, :],
                          mask=ok & mn[:, None] & (offs_rs < R // 16)[None, :], other=0.0)
            b = (_dequant_tile(bp, bsc, BLOCK_N, RC) * bg).to(tl.bfloat16)
        else:   # narrow rank blocks: gather each element's byte, nibble and scale (the join/reshape form miscompiles)
            byte = tl.load(b_base + l64 * n_s * (R // 2) + row[:, None] * (R // 2) + (offs_r // 2)[None, :],
                           mask=ok & mn[:, None] & mr[None, :], other=0)
            nib = (byte.to(tl.int32) >> ((offs_r % 2) * 4)[None, :]) & 15
            bsc = tl.load(bs_base + l64 * n_s * (R // 16) + row[:, None] * (R // 16) + (offs_r // 16)[None, :],
                          mask=ok & mn[:, None] & mr[None, :], other=0.0)
            b = (_e2m1_value(nib) * bsc.to(tl.float32) * bg).to(tl.bfloat16)
    else:
        if B_FP8:   # FP8-stored B: E4M3 codes, one fp32 scale per output row
            b_base = _ptr16(B_PTRS, s, tl.float8e4nv)
        else:
            b_base = _ptr16(B_PTRS, s, tl.bfloat16)
        b = tl.load(b_base + l64 * n_s * R + row[:, None] * R + offs_r[None, :], mask=ok & mn[:, None] & mr[None, :],
                    other=0.0)
        if B_FP8:
            sb_base = _ptr16(SB_PTRS, s, tl.float32)
            sbv = tl.load(sb_base + l64 * n_s + row, mask=ok & mn, other=0.0)
            b = (b.to(tl.float32) * sbv[:, None]).to(tl.bfloat16)
    return b


@triton.jit
def _prefetch_b(B_PTRS, SB_PTRS, s, lora, row0, n_s, R, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
                B_FP8: tl.constexpr, B_FP4: tl.constexpr):
    """L2 prefetch of adapter `lora`'s B rows [row0, row0 + BLOCK_N) of slice s (contiguous in every storage format) and of
    their scales. R == BLOCK_R (a power of two), so every address lies inside the tile."""
    l64 = lora.to(tl.int64)
    r0 = l64 * n_s + row0
    b_base = _ptr16(B_PTRS, s, tl.uint8)
    if B_FP4:
        NB: tl.constexpr = max(1, BLOCK_N * BLOCK_R // 256)
        _prefetch_l2(b_base + r0 * (R // 2) + tl.arange(0, NB) * 128)
        bs_base = _ptr16(SB_PTRS, s, tl.uint8)
        NS: tl.constexpr = max(1, BLOCK_N * BLOCK_R // 2048)
        _prefetch_l2(bs_base + r0 * (R // 16) + tl.arange(0, NS) * 128)
    elif B_FP8:
        NB: tl.constexpr = max(1, BLOCK_N * BLOCK_R // 128)
        _prefetch_l2(b_base + r0 * R + tl.arange(0, NB) * 128)
        sb_base = _ptr16(SB_PTRS, s, tl.uint8)
        NS: tl.constexpr = max(1, BLOCK_N * 4 // 128)
        _prefetch_l2(sb_base + r0 * 4 + tl.arange(0, NS) * 128)
    else:
        NB: tl.constexpr = max(1, BLOCK_N * BLOCK_R * 2 // 128)
        _prefetch_l2(b_base + r0 * (R * 2) + tl.arange(0, NB) * 128)


@triton.jit
def _lora_term(HP, B_PTRS, SB_PTRS, BG, tok, first, b_first, s, n_s0, n_s, offs_m, offs_n, mm, mn, M, R, S, L,
               SPLIT_K1: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_R: tl.constexpr,
               RC: tl.constexpr, B_PREFETCH: tl.constexpr, B_PIPE: tl.constexpr, B_FP8: tl.constexpr,
               B_FP4: tl.constexpr, DOT_F32: tl.constexpr):
    """sum over the adapters l present in the tile of [token on l] (B_l h^T), fp32 [BLOCK_N, BLOCK_M]; the tile has at least
    one adapter token (first = its smallest adapter id). h = the shrink's split-K partials summed in split order, one bf16
    rounding; the expansion bf16 x bf16 accumulated in fp32. The loop visits each distinct adapter of the tile once, and
    issues the next adapter's B loads before the current adapter's dot."""
    lo = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    hm = mm & (tok >= 0)
    for rc in tl.static_range(0, BLOCK_R, RC):
        offs_r = rc + tl.arange(0, RC)
        mr = offs_r < R
        h = tl.zeros((BLOCK_M, RC), dtype=tl.float32)
        for j in tl.static_range(SPLIT_K1):   # every partial's load in flight at once, summed in split order
            h += tl.load(HP + j * (S * M * R) + s * (M * R) + offs_m[:, None] * R + offs_r[None, :],
                         mask=hm[:, None] & mr[None, :], other=0.0)
        hb = h.to(tl.bfloat16)
        if B_PREFETCH == 2:
            if RC == BLOCK_R:
                bt = b_first          # loaded before the mainloop
            else:
                bt = _b_tile(B_PTRS, SB_PTRS, BG, s, first, first >= 0, rc, offs_n, mn, n_s0, n_s, R, L,
                             BLOCK_N, RC, B_FP8, B_FP4)
        else:
            bt = _b_tile(B_PTRS, SB_PTRS, BG, s, first, first >= 0, rc, offs_n, mn, n_s0, n_s, R, L,
                         BLOCK_N, RC, B_FP8, B_FP4)
        cur = first
        rem = (tok >= 0) & (tok != first)
        go = first >= 0
        while go:
            nxt = tl.min(tl.where(rem, tok, _BIG), axis=0)
            more = nxt < _BIG
            if B_PIPE:   # the next adapter's tile is in flight during this adapter's dot (two tiles live)
                bn = _b_tile(B_PTRS, SB_PTRS, BG, s, tl.where(more, nxt, 0), more, rc, offs_n, mn, n_s0, n_s, R, L,
                             BLOCK_N, RC, B_FP8, B_FP4)
            ht = tl.trans(tl.where((tok == cur)[:, None], hb, 0.0).to(tl.bfloat16))   # [RC, BLOCK_M]
            if DOT_F32:   # CPU interpreter (its bf16 dots are wrong)
                lo += tl.dot(bt.to(tl.float32), ht.to(tl.float32))
            else:
                lo = tl.dot(bt, ht, lo)
            if B_PIPE:
                bt = bn
            else:        # one tile live (registers); the L2 prefetch before the mainloop shortens this load
                bt = _b_tile(B_PTRS, SB_PTRS, BG, s, tl.where(more, nxt, 0), more, rc, offs_n, mn, n_s0, n_s, R, L,
                             BLOCK_N, RC, B_FP8, B_FP4)
            rem = rem & (tok != nxt)
            cur = nxt
            go = more
    return lo


# ----------------------------------------------------------------------------------------------------- kernel bodies
@triton.jit
def _quant_rows(X, XQ, SX, pid, G, M, K, stride_xm, stride_xqm,
                QBLOCK: tl.constexpr, KQ: tl.constexpr, QRN: tl.constexpr):
    """Rows pid, pid + G, ...: each token's amax over its full row, its scale and its E4M3 codes, K1's arithmetic, one row
    per pass in registers (vLLM's per-token kernel structure)."""
    offs = tl.arange(0, QBLOCK)
    for row in range(pid, M, G):
        xrow = X + row * stride_xm                    # (int32: M * K stays below 2^31 at vLLM's batch sizes)
        amax = tl.zeros((QBLOCK,), dtype=tl.float32)
        for k0 in tl.static_range(0, KQ, QBLOCK):
            xv = tl.load(xrow + k0 + offs, mask=(k0 + offs) < K, other=0.0)
            amax = tl.maximum(amax, tl.abs(xv.to(tl.float32)))
        sx = tl.maximum(tl.math.div_rn(tl.max(amax, axis=0), _FP8_MAX), _MIN_SCALE)
        qrow = XQ + row * stride_xqm
        for k0 in tl.static_range(0, KQ, QBLOCK):
            xv = tl.load(xrow + k0 + offs, mask=(k0 + offs) < K, other=0.0)
            tl.store(qrow + k0 + offs, _quant_e4m3(xv.to(tl.float32), sx, QRN), mask=(k0 + offs) < K)
        tl.store(SX + row, sx)


@triton.jit
def _shrink_seg(X, A_PTRS, SA_PTRS, AG, HP, SORTED, SEG_IDS, SEG_N, SEG_START, pid, seg,
                M, K, R, S, L, stride_xm, stride_al, stride_am,
                SPLIT_K: tl.constexpr, N_SR: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr,
                BLOCK_K: tl.constexpr, DOT_F32: tl.constexpr, A_FP8: tl.constexpr, A_FP4: tl.constexpr):
    """One program of K1's adapter shrink (fp8_lora_vllm._shrink_vllm, adapter segments; the same code, without early
    returns so that a merged kernel can signal after it): h partial pid_k of the tokens of segment seg, rank block pid_r."""
    pid_k = pid % SPLIT_K
    pid_r = (pid // SPLIT_K) % N_SR
    pid_m = pid // (SPLIT_K * N_SR)
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    lora = tl.load(SEG_IDS + seg)
    seg_n = tl.load(SEG_N + seg)
    if (lora >= 0) & (pid_m * BLOCK_M < seg_n):
        seg_start = tl.load(SEG_START + seg)
        offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mrow = offs < seg_n
        rows = tl.load(SORTED + seg_start + offs, mask=mrow, other=0)
        c0 = pid_r * BLOCK_SR
        s = c0 // R                                                        # the slice of this column block
        offs_c = (c0 - s * R) + tl.arange(0, BLOCK_SR)                      # rank rows within the slice
        if A_FP8:
            a_base = _ptr16(A_PTRS, s, tl.float8e4nv) + lora.to(tl.int64) * stride_al
        elif A_FP4:
            a_base = _ptr16(A_PTRS, s, tl.uint8) + lora.to(tl.int64) * (R * (K // 2))
            as_base = _ptr16(SA_PTRS, s, tl.float8e4nv) + lora.to(tl.int64) * (R * (K // 16))
        else:
            a_base = _ptr16(A_PTRS, s, tl.bfloat16) + lora.to(tl.int64) * stride_al
        acc = tl.zeros((BLOCK_M, BLOCK_SR), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mk = offs_k < k_hi
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=mrow[:, None] & mk[None, :],
                        other=0.0)
            if A_FP4:
                offs_kp = k0 // 2 + tl.arange(0, BLOCK_K // 2)
                offs_ks = k0 // 16 + tl.arange(0, BLOCK_K // 16)
                ap = tl.load(a_base + offs_c[:, None] * (K // 2) + offs_kp[None, :],
                             mask=(offs_kp < k_hi // 2)[None, :], other=0)
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
            acc = acc * tl.load(_ptr16(SA_PTRS, s, tl.float32) + lora.to(tl.int64) * R
                                + offs_c)[None, :]
        elif A_FP4:
            acc = acc * tl.load(AG + s * L + lora)
        hp = HP + pid_k * (S * M * R) + s * (M * R) + rows[:, None] * R + offs_c[None, :]
        tl.store(hp, acc, mask=mrow[:, None])


@triton.jit
def _spin_acquire(PTR, target):
    """Wait until *PTR >= target (gpu-scope acquire, one thread; the CTA follows through the result's broadcast). Bounded
    (2^20 polls, about half a second): a producer that never comes yields a wrong result, which the checks report,
    instead of a hung GPU."""
    c = tl.atomic_add(PTR, 0, sem="acquire", scope="gpu")
    i = 0
    while (c < target) & (i < 1048576):
        c = tl.atomic_add(PTR, 0, sem="acquire", scope="gpu")
        i += 1


@triton.jit
def _count_and_reset(SYNC, n_waiters):
    """SYNC[0] is a counter that n_waiters programs have all waited on; each of them calls this once after its wait, and
    the last one resets SYNC[0] and SYNC[1] for the next launch."""
    d = tl.atomic_add(SYNC + 1, 1, sem="relaxed", scope="gpu")
    if d == n_waiters - 1:
        tl.atomic_xchg(SYNC + 1, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(SYNC, 0, sem="relaxed", scope="gpu")


@triton.jit
def _k2_body(X, XQ, SX, W, WP, SW, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD,
             pid, pid_k, n_k2, M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, STARGET,
             stride_xm, stride_xqm, stride_wn, stride_ym,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
             SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, X_MODE: tl.constexpr, QRN: tl.constexpr,
             AMAX_M: tl.constexpr, AMAX_BK: tl.constexpr, HAS_LORA: tl.constexpr, SPLIT_K1: tl.constexpr,
             BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr, LP: tl.constexpr, B_PREFETCH: tl.constexpr,
             B_PIPE: tl.constexpr, W_PREFETCH: tl.constexpr, W_TMA: tl.constexpr, SYNC_MODE: tl.constexpr,
             PRECISE: tl.constexpr, B_FP8: tl.constexpr, B_FP4: tl.constexpr, DOT_F32: tl.constexpr,
             DEBUG_XQ: tl.constexpr, GPU: tl.constexpr):
    """One CTA of the GEMM: output tile pid (weight rows x tokens), K-split pid_k of SPLIT_K. SYNC_MODE 0: PDL
    (griddepcontrol.wait before reading the shrink's outputs; a no-op without PDL); 1: no wait (no shrink: K2 without
    LoRA); 2: the merged kernel (acquire on SYNC[2], which the shrink programs release, STARGET of them).
    SYNC[0:2] are the quantization flag and its waiter count (X_MODE 1, QTARGET releases), SYNC[2:4] the shrink's."""
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

    # (a) this CTA's weight stream starts first: its first W_PREFETCH K-steps into L2, no registers held
    if W_PREFETCH > 0:
        if GPU:
            WL: tl.constexpr = W_PREFETCH * BLOCK_K // 128
            pr = tl.minimum(offs_n, N - 1).to(tl.int64)
            pk = tl.minimum(k_lo + tl.arange(0, WL) * 128, tl.maximum(k_hi, k_lo + 1) - 1)   # this split only
            _prefetch_l2(WP + pr[:, None] * stride_wn + pk[None, :])

    # (b) the tile's adapters, before the mainloop: vLLM's metadata in one vector load (rows past the segments' total are
    # CUDA-graph padding, and the token map is stale when no token has an adapter: no adapter then), the first adapter's
    # B tile into registers, the others' into L2
    RC: tl.constexpr = min(BLOCK_R, R_CHUNK)
    if HAS_LORA:
        offs_l = tl.arange(0, LP)
        segn = tl.load(SEG_N + offs_l, mask=offs_l < L1, other=0)
        segi = tl.load(SEG_IDS + offs_l, mask=offs_l < L1, other=-1)
        n_all = tl.sum(segn, axis=0)
        n_ad = tl.sum(tl.where(segi >= 0, segn, 0), axis=0)
        tok = tl.load(TOK + offs_m, mask=mm & (offs_m < n_all) & (n_ad > 0), other=-1)
        any_ad = tl.max(tok, axis=0) >= 0
        first = tl.min(tl.where(tok >= 0, tok, _BIG), axis=0)
        first = tl.where(any_ad, first, -1)
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32) + (n0 >= END2).to(tl.int32)   # up to 4 slices
        n_s0 = tl.where(s == 0, 0, tl.where(s == 1, END0, tl.where(s == 2, END1, END2)))
        n_s = tl.where(s == 0, END0, tl.where(s == 1, END1 - END0, tl.where(s == 2, END2 - END1, N - END2)))
        b_first = tl.zeros((BLOCK_N, RC), dtype=tl.bfloat16)
        if B_PREFETCH == 2:
            if RC == BLOCK_R:
                b_first = _b_tile(B_PTRS, SB_PTRS, BG, s, tl.maximum(first, 0), any_ad, 0, offs_n, mn, n_s0, n_s, R,
                                  L1 - 1, BLOCK_N, RC, B_FP8, B_FP4)
        if B_PREFETCH >= 1:
            if GPU:
                if B_PREFETCH == 2:
                    pf = (tok >= 0) & (tok != first)
                else:
                    pf = tok >= 0
                pgo = tl.max(pf.to(tl.int32), axis=0) > 0
                while pgo:
                    pl = tl.min(tl.where(pf, tok, _BIG), axis=0)
                    _prefetch_b(B_PTRS, SB_PTRS, s, pl, n0 - n_s0, n_s, R, BLOCK_N, BLOCK_R, B_FP8, B_FP4)
                    pf = pf & (tok != pl)
                    pgo = tl.max(pf.to(tl.int32), axis=0) > 0

    # (c) the activations' per-token scales
    if X_MODE == 0:   # each token's amax over its full row (exact in any order), K1's scale rule
        ar = tl.arange(0, BLOCK_M)
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for mr0 in range(0, BLOCK_M, AMAX_M):
            if m0 + mr0 < M:
                rows = m0 + mr0 + tl.arange(0, AMAX_M)
                mrow = rows < M
                a = tl.zeros((AMAX_M,), dtype=tl.float32)
                for ka in range(0, K, AMAX_BK):
                    offs_a = ka + tl.arange(0, AMAX_BK)
                    xa = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_a[None, :],
                                 mask=mrow[:, None] & (offs_a < K)[None, :], other=0.0)
                    a = tl.maximum(a, tl.max(tl.abs(xa.to(tl.float32)), axis=1))
                hit = ar[:, None] == (mr0 + tl.arange(0, AMAX_M))[None, :]
                amax = tl.maximum(amax, tl.max(tl.where(hit, a[None, :], 0.0), axis=1))
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
    else:
        if X_MODE == 1:   # the shrink kernel's quantization programs release SYNC[0] once each
            _spin_acquire(SYNC, QTARGET)
            _count_and_reset(SYNC, n_k2)
        else:             # X_MODE 2: the whole shrink kernel first
            if SYNC_MODE == 0:
                if GPU:
                    tl.extra.cuda.gdc_wait()
            elif SYNC_MODE == 2:
                _spin_acquire(SYNC + 2, STARGET)
        sx = tl.load(SX + offs_m, mask=mm, other=1.0, cache_modifier=".cg")

    # (d) mainloop: the weight stream (swap-AB: W rows on wgmma's 64-row side, tokens on its narrow side)
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if W_TMA:
            w = W.load([n0, k0])
        elif EVEN_K:
            w = tl.load(W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=mn[:, None], other=0.0)
        else:
            w = tl.load(W + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :],
                        mask=mn[:, None] & (offs_k < k_hi)[None, :], other=0.0)
        if EVEN_K:
            mx = mm[None, :]
        else:
            mx = mm[None, :] & (offs_k < k_hi)[:, None]
        if X_MODE == 0:   # quantize this K-tile of x on the fly, K1's arithmetic
            xb = tl.load(X + offs_m[None, :].to(tl.int64) * stride_xm + offs_k[:, None], mask=mx, other=0.0)
            xt = _quant_e4m3(xb.to(tl.float32), sx[None, :], QRN)
            if DEBUG_XQ:
                if pid_n == 0:
                    tl.store(XQD + offs_m[None, :].to(tl.int64) * K + offs_k[:, None], xt, mask=mx)
        else:
            xt = tl.load(XQ + offs_m[None, :].to(tl.int64) * stride_xqm + offs_k[:, None], mask=mx, other=0.0,
                         cache_modifier=".cg")
        if DOT_F32:   # CPU interpreter only
            acc = tl.dot(w.to(tl.float32), xt.to(tl.float32), acc)
        elif PRECISE:   # FP8 MMA accumulates with reduced precision: promote to fp32 after every K-block, as CUTLASS does
            acc += tl.dot(w, xt)
        else:
            acc = tl.dot(w, xt, acc)

    # (e) the shrink is needed only now
    if SYNC_MODE == 0:
        if X_MODE != 2:
            if GPU:
                tl.extra.cuda.gdc_wait()
    elif SYNC_MODE == 2:
        if X_MODE != 2:
            _spin_acquire(SYNC + 2, STARGET)
        _count_and_reset(SYNC + 2, n_k2)
    scale = sx[None, :] * tl.load(SW)
    msk = mm[None, :] & mn[:, None]
    yp = Y + offs_m[None, :].to(tl.int64) * stride_ym + offs_n[:, None]
    if SPLIT_K == 1:
        out = acc * scale
        if HAS_LORA:
            if any_ad:
                out += _lora_term(HP, B_PTRS, SB_PTRS, BG, tok, first, b_first, s, n_s0, n_s, offs_m, offs_n, mm, mn,
                                  M, R, S, L1 - 1, SPLIT_K1, BLOCK_M, BLOCK_N, BLOCK_R, RC, B_PREFETCH, B_PIPE, B_FP8,
                                  B_FP4, DOT_F32)
        tl.store(yp, out.to(tl.bfloat16), mask=msk)
    else:   # fp32 partial; the last CTA of this tile to arrive reduces all of them in split order
        pp = P + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(pp + pid_k.to(tl.int64) * M * N, acc, mask=msk)
        tl.debug_barrier()                                            # every thread's partial before the release
        old = tl.atomic_add(CNT + pid, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for j in tl.static_range(SPLIT_K):
                pj = tl.load(pp + j * M * N, mask=msk & (pid_k != j), other=0.0, cache_modifier=".cg")
                tot += tl.where(pid_k == j, acc, pj)
            out = tot * scale
            if HAS_LORA:
                if any_ad:
                    out += _lora_term(HP, B_PTRS, SB_PTRS, BG, tok, first, b_first, s, n_s0, n_s, offs_m, offs_n, mm,
                                      mn, M, R, S, L1 - 1, SPLIT_K1, BLOCK_M, BLOCK_N, BLOCK_R, RC, B_PREFETCH, B_PIPE,
                                      B_FP8, B_FP4, DOT_F32)
            tl.store(yp, out.to(tl.bfloat16), mask=msk)
            tl.atomic_xchg(CNT + pid, 0, sem="relaxed", scope="gpu")   # ready for the next launch


# ----------------------------------------------------------------------------------------------------------- kernels
@triton.jit(do_not_specialize=["M", "S", "L1", "NX"])
def _shrink_pdl(X, A_PTRS, SA_PTRS, AG, HP, XQ, SX, SYNC, SORTED, SEG_IDS, SEG_N, SEG_START,
                M, K, R, S, L1, NX, stride_xm, stride_al, stride_am, stride_xqm,
                SPLIT_K: tl.constexpr, N_SR: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr,
                BLOCK_K: tl.constexpr, QSEG: tl.constexpr, QSIGNAL: tl.constexpr, QBLOCK: tl.constexpr,
                KQ: tl.constexpr, QRN: tl.constexpr, USE_GDC: tl.constexpr, DOT_F32: tl.constexpr = False,
                A_FP8: tl.constexpr = False, A_FP4: tl.constexpr = False):
    """grid (NX', 1, L1 [+ 1]): axis 2 = vLLM's L1 = L + 1 adapter segments (the shrink; programs with pid >= NX have no
    work there), preceded with QSEG by one segment of per-token quantization programs (all NX' of them release SYNC[0]
    with QSIGNAL). Every program first lets the dependent GEMM launch."""
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    seg = tl.program_id(2)
    if QSEG:
        if seg == 0:
            _quant_rows(X, XQ, SX, pid, tl.num_programs(0), M, K, stride_xm, stride_xqm, QBLOCK, KQ, QRN)
            if QSIGNAL:
                tl.debug_barrier()
                tl.atomic_add(SYNC, 1, sem="release", scope="gpu")
        elif pid < NX:
            _shrink_seg(X, A_PTRS, SA_PTRS, AG, HP, SORTED, SEG_IDS, SEG_N, SEG_START, pid, seg - 1,
                        M, K, R, S, L1 - 1, stride_xm, stride_al, stride_am,
                        SPLIT_K, N_SR, BLOCK_M, BLOCK_SR, BLOCK_K, DOT_F32, A_FP8, A_FP4)
    else:
        if pid < NX:
            _shrink_seg(X, A_PTRS, SA_PTRS, AG, HP, SORTED, SEG_IDS, SEG_N, SEG_START, pid, seg,
                        M, K, R, S, L1 - 1, stride_xm, stride_al, stride_am,
                        SPLIT_K, N_SR, BLOCK_M, BLOCK_SR, BLOCK_K, DOT_F32, A_FP8, A_FP4)


@triton.jit(do_not_specialize=["M", "S", "L1", "END0", "END1", "END2", "QTARGET", "STARGET"])
def _gemm_pdl(X, XQ, SX, W, WP, SW, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD,
              M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, STARGET,
              stride_xm, stride_xqm, stride_wn, stride_ym,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
              SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, X_MODE: tl.constexpr, QRN: tl.constexpr,
              AMAX_M: tl.constexpr, AMAX_BK: tl.constexpr, HAS_LORA: tl.constexpr, SPLIT_K1: tl.constexpr,
              BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr, LP: tl.constexpr, B_PREFETCH: tl.constexpr,
              B_PIPE: tl.constexpr, W_PREFETCH: tl.constexpr, W_TMA: tl.constexpr, SYNC_MODE: tl.constexpr,
              PRECISE: tl.constexpr, B_FP8: tl.constexpr = False, B_FP4: tl.constexpr = False, DOT_F32: tl.constexpr = False,
              DEBUG_XQ: tl.constexpr = False, GPU: tl.constexpr = True):
    """grid (output tiles, SPLIT_K); launched with launch_pdl=True after _shrink_pdl (SYNC_MODE 0)."""
    _k2_body(X, XQ, SX, W, WP, SW, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD,
             tl.program_id(0), tl.program_id(1), tl.num_programs(0) * tl.num_programs(1),
             M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, STARGET,
             stride_xm, stride_xqm, stride_wn, stride_ym,
             BLOCK_M, BLOCK_N, BLOCK_K, GROUP_N, SPLIT_K, EVEN_K, X_MODE, QRN, AMAX_M, AMAX_BK, HAS_LORA, SPLIT_K1,
             BLOCK_R, R_CHUNK, LP, B_PREFETCH, B_PIPE, W_PREFETCH, W_TMA, SYNC_MODE, PRECISE, B_FP8, B_FP4, DOT_F32,
             DEBUG_XQ, GPU)


@triton.jit(do_not_specialize=["M", "S", "L1", "END0", "END1", "END2", "QTARGET", "STARGET", "NSH", "NSX"])
def _coop(X, XQ, SX, W, WP, SW, HP, A_PTRS, SA_PTRS, AG, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SORTED, SEG_IDS,
          SEG_N, SEG_START, XQD,
          M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, STARGET, NSH, NSX, NT,
          stride_xm, stride_xqm, stride_wn, stride_ym, stride_al, stride_am,
          S_SPLIT_K: tl.constexpr, N_SR: tl.constexpr, S_BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr,
          S_BLOCK_K: tl.constexpr, QBLOCK: tl.constexpr, KQ: tl.constexpr,
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
          SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, X_MODE: tl.constexpr, QRN: tl.constexpr,
          AMAX_M: tl.constexpr, AMAX_BK: tl.constexpr, SPLIT_K1: tl.constexpr,
          BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr, LP: tl.constexpr, B_PREFETCH: tl.constexpr,
          B_PIPE: tl.constexpr, W_PREFETCH: tl.constexpr, W_TMA: tl.constexpr, PRECISE: tl.constexpr,
          A_FP8: tl.constexpr = False, A_FP4: tl.constexpr = False, B_FP8: tl.constexpr = False,
          B_FP4: tl.constexpr = False, DOT_F32: tl.constexpr = False, DEBUG_XQ: tl.constexpr = False,
          GPU: tl.constexpr = True):
    """The fallback: one 1-D grid. Programs [0, NSH) run the shrink (NSX programs per segment; with X_MODE 1 the first NSX
    are the per-token quantization programs, which release SYNC[0]) and release SYNC[2] after it; programs [NSH, NSH + NT
    * SPLIT_K) run the GEMM (SYNC_MODE 2: acquire SYNC[2] == STARGET = NSH before the epilogue)."""
    pid = tl.program_id(0)
    if pid < NSH:
        seg = pid // NSX
        sp = pid % NSX
        if X_MODE == 0:
            _shrink_seg(X, A_PTRS, SA_PTRS, AG, HP, SORTED, SEG_IDS, SEG_N, SEG_START, sp, seg,
                        M, K, R, S, L1 - 1, stride_xm, stride_al, stride_am,
                        S_SPLIT_K, N_SR, S_BLOCK_M, BLOCK_SR, S_BLOCK_K, DOT_F32, A_FP8, A_FP4)
        else:
            if seg == 0:
                _quant_rows(X, XQ, SX, sp, NSX, M, K, stride_xm, stride_xqm, QBLOCK, KQ, QRN)
                if X_MODE == 1:
                    tl.debug_barrier()
                    tl.atomic_add(SYNC, 1, sem="release", scope="gpu")
            else:
                _shrink_seg(X, A_PTRS, SA_PTRS, AG, HP, SORTED, SEG_IDS, SEG_N, SEG_START, sp, seg - 1,
                            M, K, R, S, L1 - 1, stride_xm, stride_al, stride_am,
                            S_SPLIT_K, N_SR, S_BLOCK_M, BLOCK_SR, S_BLOCK_K, DOT_F32, A_FP8, A_FP4)
        tl.debug_barrier()
        tl.atomic_add(SYNC + 2, 1, sem="release", scope="gpu")
    else:
        q = pid - NSH
        _k2_body(X, XQ, SX, W, WP, SW, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD,
                 q % NT, q // NT, NT * SPLIT_K, M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, STARGET,
                 stride_xm, stride_xqm, stride_wn, stride_ym,
                 BLOCK_M, BLOCK_N, BLOCK_K, GROUP_N, SPLIT_K, EVEN_K, X_MODE, QRN, AMAX_M, AMAX_BK, True, SPLIT_K1,
                 BLOCK_R, R_CHUNK, LP, B_PREFETCH, B_PIPE, W_PREFETCH, W_TMA, 2, PRECISE, B_FP8, B_FP4, DOT_F32,
                 DEBUG_XQ, GPU)


# ------------------------------------------------------------------------------------------------------------- host
def default_cfg(M: int, N: int, K: int, widths, R: int, lora_fp4: bool = False) -> dict:
    """Default tiles (placeholders until the H100 sweep, bench/pdl_k2_bench.py): 16-token tiles, 64 weight rows (128 when
    there are many N-tiles), split-K until the projection has about 2 CTAs per SM, the shrink in 4 K-splits."""
    bm = 16 if M <= 16 else 32 if M <= 32 else 64
    bn = 64
    while any(w % bn for w in widths) and bn > 16:
        bn //= 2
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = 1
    while tiles * sk < 2 * 132 and K // (sk * 2) >= 2 * 256 and sk < 8:
        sk *= 2
    br = max(16, triton.next_power_of_2(R))
    # 8 warps (two warpgroups) from 4096-element output tiles: at 4 warps BLOCK_M 64 spilled (255 registers + 16 bytes)
    return {"X_MODE": 0 if M <= 2 else 1, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 256 if M <= 32 else 128,
            "SPLIT_K": sk, "num_warps": 8 if bm * bn >= 4096 else 4, "num_stages": 4, "GROUP_N": 8, "W_TMA": False,
            "W_PREFETCH": 2,
            "B_PREFETCH": 2 if (bn * br <= 8192 and not (lora_fp4 and br > 64)) else 1,
            "SPLIT_K1": 4 if K <= 4096 else 8, "S_BLOCK_K": 128, "S_STAGES": 2}


class _Work:
    """Per-device buffers shared by every PDL op: split-K partials (grown before any capture), per-tile arrival counters
    and the four synchronization words (each launch returns every counter to zero), a 1-byte dummy for unused pointers."""

    def __init__(self, device):
        self.device = device
        self.p = torch.empty(0, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(0, dtype=torch.int32, device=device)
        self.sync = torch.zeros(4, dtype=torch.int32, device=device)
        self.keep = []                                  # replaced buffers stay alive: captured graphs may point at them

    def ensure(self, n_p: int, n_cnt: int) -> None:
        if self.p.numel() < n_p:
            self.keep.append(self.p)
            self.p = torch.empty(n_p, dtype=torch.float32, device=self.device)
        if self.cnt.numel() < n_cnt:
            self.keep.append(self.cnt)
            self.cnt = torch.zeros(n_cnt, dtype=torch.int32, device=self.device)

    def reset(self) -> None:
        """Zero the counters (only needed after a launch that did not complete)."""
        self.cnt.zero_()
        self.sync.zero_()


_WORK: dict = {}


def work(device) -> _Work:
    key = str(device)
    if key not in _WORK:
        _WORK[key] = _Work(device)
    return _WORK[key]
