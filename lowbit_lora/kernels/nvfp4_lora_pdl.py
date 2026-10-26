"""Blackwell (sm_120) decode path with the LoRA adapters off the critical path, for NVFP4 (W4A4) bases: a shrink kernel
overlapped, through programmatic dependent launch (PDL), with a block-scaled FP4 GEMM that quantizes its own activations
and adds every adapter's expansion in its epilogue. The NVFP4 counterpart of fp8_lora_pdl.py (HOPPER-PDL-K2's design).

The plugin's NVFP4 fused path (nvfp4_lora_vllm.py) runs K1 (x's NVFP4 codes and block scales, every adapter's shrink) and
then K2 (tl.dot_scaled with e2m1 operands, the LoRA expansion in its epilogue). As on the H100, K2 cannot start before K1
has finished and its epilogue walks vLLM's segment metadata one segment at a time. Here:

  shrink   K1's adapter segments (fp8_lora_pdl._shrink_seg: the same code as nvfp4_lora_vllm._shrink_nvfp4's), launched
           first; every program executes griddepcontrol.launch_dependents at its start. With X_MODE 1 its grid has one more
           segment of quantization programs: K1's all-token pass (nvfp4_lora_vllm._quant_nvfp4, vLLM's scaled_fp4_quant
           recipe), each program a K-slice of a row tile, which release a counter.
  GEMM     launched with launch_pdl=True; needs nothing from the shrink before its epilogue:
             X_MODE 0  it quantizes its own activations: NVFP4 block scales are local to 16 elements and the global
                       multiplier is static, so each K-tile of x is quantized on the fly with K1's own function
                       (_quant_nvfp4: E4M3 block scale of gs * blockmax / 6, rcp.approx.ftz as vLLM, saturating E2M1
                       round-to-nearest) - bit-identical to K1's codes and scales whatever the tiling;
             X_MODE 1  it reads the codes and scales of the quantization programs, after acquiring their counter.
           Before its mainloop: vLLM's metadata in one vector load, the tile's adapters, the first adapter's B tile
           (dequantized) in registers, the others' and its own first weight tiles (codes and block scales) in L2. It
           streams the FP4 weights while the shrink runs and executes griddepcontrol.wait only before its epilogue reads h.
           SWAP: the weight rows on the MMA's 16-row side and the tokens on its 8-wide side (tiles of 8 tokens), or the
           plugin's orientation (tokens on the 16-row side); both lower to sm_120's warp-level block-scaled MMA
           (mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X).
  split-K  as fp8_lora_pdl: fp32 partials, a per-tile arrival counter, the last CTA sums them in split order
           (deterministic), applies alpha once, adds the LoRA term, stores bf16.

Arithmetic is the NVFP4 fused path's: block-scaled FP4 products accumulated in fp32 by the MMA, alpha = 1 / (x_global *
w_global) applied once, h summed over the shrink's split partials in split order and rounded to bf16 once, the expansion
as bf16 products accumulated in fp32, one rounding to bf16.

CPU interpreter (tests/pdl_fp4_interp_check.py): GPU=False drops the griddepcontrol and prefetch instructions, PTX=False
replaces K1's inline-PTX conversions (rcp.approx, RN E4M3) by exact arithmetic (as nvfp4_lora_vllm), DOT_F32 decodes both
operands for an fp32 dot (the interpreter has no dot_scaled). sm_120 compile: tests/pdl_fp4_interp_check.py --compile.
"""
from __future__ import annotations

import triton
import triton.language as tl

from lowbit_lora.kernels.fp8_lora_pdl import (_BIG, _b_tile, _count_and_reset, _lora_term, _prefetch_b, _prefetch_l2,
                                              _shrink_seg, _spin_acquire, work)
from lowbit_lora.kernels.nvfp4_lora_vllm import _dequant_tile, _quant_nvfp4

__all__ = ["_shrink_pdl4", "_gemm_pdl4", "default_cfg4", "warps4", "work"]


@triton.jit
def _quant_tiles4(X, XQ, SXQ, GS, pid, G, M, K, stride_xm, stride_xqm, stride_sxm,
                  QBM: tl.constexpr, QBK: tl.constexpr, QSPLIT: tl.constexpr, PTX: tl.constexpr):
    """Quantization programs pid, pid + G, ...: program q quantizes rows [(q // QSPLIT) * QBM, +QBM) over its K-slice
    q % QSPLIT, in QBK-wide tiles, with K1's all-token code (nvfp4_lora_vllm._shrink_nvfp4, segment L1)."""
    gs = tl.load(GS)
    n_q = tl.cdiv(M, QBM) * QSPLIT
    k_slice = tl.cdiv(tl.cdiv(K, QSPLIT), QBK) * QBK
    for q in range(pid, n_q, G):
        rows = (q // QSPLIT) * QBM + tl.arange(0, QBM)
        mrow = rows < M
        q_lo = (q % QSPLIT) * k_slice
        q_hi = tl.minimum(q_lo + k_slice, K)
        for k0 in range(q_lo, q_hi, QBK):
            offs_k = k0 + tl.arange(0, QBK)
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
                        mask=mrow[:, None] & (offs_k[None, :] < q_hi), other=0.0).to(tl.float32)
            packed, sf = _quant_nvfp4(x, gs, QBM, QBK, PTX)
            offs_p = k0 // 2 + tl.arange(0, QBK // 2)
            tl.store(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_p[None, :], packed,
                     mask=mrow[:, None] & (offs_p[None, :] < q_hi // 2))
            offs_s = k0 // 16 + tl.arange(0, QBK // 16)
            tl.store(SXQ + rows[:, None].to(tl.int64) * stride_sxm + offs_s[None, :], sf,
                     mask=mrow[:, None] & (offs_s[None, :] < q_hi // 16))


@triton.jit(do_not_specialize=["M", "S", "L1", "NX"])
def _shrink_pdl4(X, A_PTRS, SA_PTRS, AG, HP, XQ, SXQ, GS, SYNC, SORTED, SEG_IDS, SEG_N, SEG_START,
                 M, K, R, S, L1, NX, stride_xm, stride_al, stride_am, stride_xqm, stride_sxm,
                 SPLIT_K: tl.constexpr, N_SR: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr,
                 BLOCK_K: tl.constexpr, QSEG: tl.constexpr, QSIGNAL: tl.constexpr, QBM: tl.constexpr,
                 QBK: tl.constexpr, QSPLIT: tl.constexpr, PTX: tl.constexpr, USE_GDC: tl.constexpr,
                 DOT_F32: tl.constexpr = False, A_FP8: tl.constexpr = False, A_FP4: tl.constexpr = False):
    """grid (NX', 1, L1 [+ 1]): axis 2 = vLLM's L1 = L + 1 adapter segments (the shrink; programs with pid >= NX have no
    work there), preceded with QSEG by one segment of NVFP4 quantization programs (all NX' of them release SYNC[0] with
    QSIGNAL). Every program first lets the dependent GEMM launch."""
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    pid = tl.program_id(0)
    seg = tl.program_id(2)
    if QSEG:
        if seg == 0:
            _quant_tiles4(X, XQ, SXQ, GS, pid, tl.num_programs(0), M, K, stride_xm, stride_xqm, stride_sxm,
                          QBM, QBK, QSPLIT, PTX)
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


@triton.jit
def _prefetch_w4(W, SW, offs_n, k_lo, k_hi, N, K, WL: tl.constexpr, WSL: tl.constexpr):
    """L2 prefetch of this CTA's first weight tiles: codes [rows, k/2 bytes] and E4M3 block scales [rows, k/16 bytes]."""
    pr = tl.minimum(offs_n, N - 1).to(tl.int64)
    top = tl.maximum(k_hi, k_lo + 2) // 2 - 1
    pk = tl.minimum(k_lo // 2 + tl.arange(0, WL) * 128, top)
    _prefetch_l2(W + pr[:, None] * (K // 2) + pk[None, :])
    tops = tl.maximum(k_hi, k_lo + 16) // 16 - 1
    ps = tl.minimum(k_lo // 16 + tl.arange(0, WSL) * 128, tops)
    _prefetch_l2(SW + pr[:, None] * (K // 16) + ps[None, :])


@triton.jit
def _k2fp4_body(X, XQ, SXQ, GS, W, SW, ALPHA, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N,
                XQD, SXD, pid, pid_k, n_k2, M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET,
                stride_xm, stride_xqm, stride_sxm, stride_ym,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
                SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, SWAP: tl.constexpr, X_MODE: tl.constexpr,
                HAS_LORA: tl.constexpr, SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr,
                LP: tl.constexpr, B_PREFETCH: tl.constexpr, B_PIPE: tl.constexpr, W_PREFETCH: tl.constexpr,
                SYNC_MODE: tl.constexpr, PTX: tl.constexpr, B_FP8: tl.constexpr, B_FP4: tl.constexpr,
                DOT_F32: tl.constexpr, DEBUG_XQ: tl.constexpr, GPU: tl.constexpr):
    """One CTA of the GEMM: output tile pid (weight rows x tokens), K-split pid_k of SPLIT_K. SYNC_MODE 0: PDL
    (griddepcontrol.wait before reading h; a no-op without PDL); 1: no wait (no shrink launched: the GEMM without LoRA).
    SYNC[0:2]: the quantization programs' counter and its waiter count (X_MODE 1, QTARGET releases)."""
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

    # (a) this CTA's weight stream starts first: its first W_PREFETCH K-steps (codes and block scales) into L2
    if W_PREFETCH > 0:
        if GPU:
            _prefetch_w4(W, SW, offs_n, k_lo, k_hi, N, K, max(1, W_PREFETCH * BLOCK_K // 256),
                         max(1, W_PREFETCH * BLOCK_K // 2048))

    # (b) the tile's adapters, before the mainloop (as fp8_lora_pdl._k2_body)
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
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32) + (n0 >= END2).to(tl.int32)
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

    # (c) the activations: quantized here (X_MODE 0), or by the quantization programs (X_MODE 1: acquire their counter)
    gs = tl.load(GS)
    if X_MODE == 1:
        _spin_acquire(SYNC, QTARGET)
        _count_and_reset(SYNC, n_k2)

    # (d) mainloop: the FP4 weight stream, block-scaled MMA
    if SWAP:
        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_p = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        offs_s = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        if EVEN_K:
            mw = mn[:, None]
            mx = mm[:, None]
        else:
            mw = mn[:, None] & (offs_p < k_hi // 2)[None, :]
            mx = mm[:, None] & (offs_p < k_hi // 2)[None, :]
        if EVEN_K:
            mws = mn[:, None]
            mxs = mm[:, None]
        else:
            mws = mn[:, None] & (offs_s < k_hi // 16)[None, :]
            mxs = mm[:, None] & (offs_s < k_hi // 16)[None, :]
        wq = tl.load(W + offs_n[:, None].to(tl.int64) * (K // 2) + offs_p[None, :], mask=mw, other=0)
        ws = tl.load(SW + offs_n[:, None].to(tl.int64) * (K // 16) + offs_s[None, :], mask=mws, other=0.0)
        if X_MODE == 0:   # K1's quantization of this K-tile (block-local: identical codes and scales whatever the tiling)
            offs_k = k0 + tl.arange(0, BLOCK_K)
            if EVEN_K:
                mxk = mm[:, None]
            else:
                mxk = mm[:, None] & (offs_k < k_hi)[None, :]
            xb = tl.load(X + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=mxk,
                         other=0.0).to(tl.float32)
            xq, xs = _quant_nvfp4(xb, gs, BLOCK_M, BLOCK_K, PTX)
            if DEBUG_XQ:
                if pid_n == 0:
                    tl.store(XQD + offs_m[:, None].to(tl.int64) * (K // 2) + offs_p[None, :], xq, mask=mx)
                    tl.store(SXD + offs_m[:, None].to(tl.int64) * (K // 16) + offs_s[None, :], xs, mask=mxs)
        else:
            xq = tl.load(XQ + offs_m[:, None].to(tl.int64) * stride_xqm + offs_p[None, :], mask=mx, other=0,
                         cache_modifier=".cg")
            xs = tl.load(SXQ + offs_m[:, None].to(tl.int64) * stride_sxm + offs_s[None, :], mask=mxs, other=0.0,
                         cache_modifier=".cg")
        if DOT_F32:   # CPU interpreter: decode both operands, fp32 dot
            if SWAP:
                acc = tl.dot(_dequant_tile(wq, ws, BLOCK_N, BLOCK_K), tl.trans(_dequant_tile(xq, xs, BLOCK_M, BLOCK_K)),
                             acc)
            else:
                acc = tl.dot(_dequant_tile(xq, xs, BLOCK_M, BLOCK_K), tl.trans(_dequant_tile(wq, ws, BLOCK_N, BLOCK_K)),
                             acc)
        else:
            if SWAP:
                acc = tl.dot_scaled(wq, ws, "e2m1", tl.trans(xq), xs, "e2m1", acc)
            else:
                acc = tl.dot_scaled(xq, xs, "e2m1", tl.trans(wq), ws, "e2m1", acc)
    if SWAP:
        accn = acc
    else:
        accn = tl.trans(acc)                                            # [BLOCK_N, BLOCK_M] from here on

    # (e) the shrink is needed only now
    if SYNC_MODE == 0:
        if GPU:
            tl.extra.cuda.gdc_wait()
    alpha = tl.load(ALPHA)
    msk = mm[None, :] & mn[:, None]
    yp = Y + offs_m[None, :].to(tl.int64) * stride_ym + offs_n[:, None]
    if SPLIT_K == 1:
        out = accn * alpha
        if HAS_LORA:
            if any_ad:
                out += _lora_term(HP, B_PTRS, SB_PTRS, BG, tok, first, b_first, s, n_s0, n_s, offs_m, offs_n, mm, mn,
                                  M, R, S, L1 - 1, SPLIT_K1, BLOCK_M, BLOCK_N, BLOCK_R, RC, B_PREFETCH, B_PIPE, B_FP8,
                                  B_FP4, DOT_F32)
        tl.store(yp, out.to(tl.bfloat16), mask=msk)
    else:   # fp32 partial; the last CTA of this tile to arrive reduces all of them in split order
        pp = P + offs_m[None, :].to(tl.int64) * N + offs_n[:, None]
        tl.store(pp + pid_k.to(tl.int64) * M * N, accn, mask=msk)
        tl.debug_barrier()
        old = tl.atomic_add(CNT + pid, 1, sem="acq_rel", scope="gpu")
        if old == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for j in tl.static_range(SPLIT_K):
                pj = tl.load(pp + j * M * N, mask=msk & (pid_k != j), other=0.0, cache_modifier=".cg")
                tot += tl.where(pid_k == j, accn, pj)
            out = tot * alpha
            if HAS_LORA:
                if any_ad:
                    out += _lora_term(HP, B_PTRS, SB_PTRS, BG, tok, first, b_first, s, n_s0, n_s, offs_m, offs_n, mm,
                                      mn, M, R, S, L1 - 1, SPLIT_K1, BLOCK_M, BLOCK_N, BLOCK_R, RC, B_PREFETCH, B_PIPE,
                                      B_FP8, B_FP4, DOT_F32)
            tl.store(yp, out.to(tl.bfloat16), mask=msk)
            tl.atomic_xchg(CNT + pid, 0, sem="relaxed", scope="gpu")


@triton.jit(do_not_specialize=["M", "S", "L1", "END0", "END1", "END2", "QTARGET"])
def _gemm_pdl4(X, XQ, SXQ, GS, W, SW, ALPHA, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD, SXD,
               M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, stride_xm, stride_xqm, stride_sxm, stride_ym,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_N: tl.constexpr,
               SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr, SWAP: tl.constexpr, X_MODE: tl.constexpr,
               HAS_LORA: tl.constexpr, SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, R_CHUNK: tl.constexpr,
               LP: tl.constexpr, B_PREFETCH: tl.constexpr, B_PIPE: tl.constexpr, W_PREFETCH: tl.constexpr,
               SYNC_MODE: tl.constexpr, PTX: tl.constexpr = True, B_FP8: tl.constexpr = False,
               B_FP4: tl.constexpr = False, DOT_F32: tl.constexpr = False, DEBUG_XQ: tl.constexpr = False,
               GPU: tl.constexpr = True):
    """grid (output tiles, SPLIT_K); launched with launch_pdl=True after _shrink_pdl4 (SYNC_MODE 0)."""
    _k2fp4_body(X, XQ, SXQ, GS, W, SW, ALPHA, HP, B_PTRS, SB_PTRS, BG, Y, P, CNT, SYNC, TOK, SEG_IDS, SEG_N, XQD, SXD,
                tl.program_id(0), tl.program_id(1), tl.num_programs(0) * tl.num_programs(1),
                M, N, K, R, S, L1, END0, END1, END2, K_SPLIT, QTARGET, stride_xm, stride_xqm, stride_sxm, stride_ym,
                BLOCK_M, BLOCK_N, BLOCK_K, GROUP_N, SPLIT_K, EVEN_K, SWAP, X_MODE, HAS_LORA, SPLIT_K1, BLOCK_R,
                R_CHUNK, LP, B_PREFETCH, B_PIPE, W_PREFETCH, SYNC_MODE, PTX, B_FP8, B_FP4, DOT_F32, DEBUG_XQ, GPU)


def warps4(bm: int, bn: int, br: int) -> int:
    """sm_120 register budget (offline compile, cuobjdump): at 4 warps 128-wide weight tiles spilled (255 registers + 8-16
    bytes of stack), and so did rank 128's 64-wide epilogue chunks; 8 warps for those."""
    return 8 if (bn >= 128 or bm * bn >= 4096 or br >= 128) else 4


def default_cfg4(M: int, N: int, K: int, widths, R: int, n_sm: int = 94, lora_fp4: bool = False) -> dict:
    """Default tiles (placeholders until the sm_120 sweep, bench/pdl_fp4_bench.py): swap-AB with 8-token tiles up to 8
    tokens and 16 up to 16 (32 / 64 above, the plugin's orientation), 64 weight rows, split-K until the projection has
    about one CTA per SM, x quantized once by the quantization programs (X_MODE 1, HOPPER-PDL-K2's winner)."""
    swap = M <= 16
    bm = (8 if M <= 8 else 16) if swap else (16 if M <= 16 else 32 if M <= 32 else 64)
    bn = 64
    while any(w % bn for w in widths) and bn > 16:
        bn //= 2
    tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    sk = 1
    while tiles * sk < n_sm and K // (sk * 2) >= 2 * 256 and sk < 8:
        sk *= 2
    br = max(16, triton.next_power_of_2(R))
    return {"X_MODE": 1, "SWAP": swap, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 256, "SPLIT_K": sk,
            "num_warps": warps4(bm, bn, br), "num_stages": 4, "GROUP_N": 8, "W_PREFETCH": 2,
            "B_PREFETCH": 2 if (bn * br <= 8192 and not (lora_fp4 and br > 64)) else 1,
            "SPLIT_K1": 4 if K <= 4096 else 8, "S_BLOCK_K": 128, "S_STAGES": 2, "QBM": 16, "QBK": 256}
