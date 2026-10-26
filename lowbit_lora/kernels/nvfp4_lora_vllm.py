"""Fused NVFP4 (W4A4) GEMM + multi-adapter LoRA on vLLM's LoRA buffers, for Blackwell sm_120 (Phase 1).

The NVFP4 counterpart of fp8_lora_vllm.py, in vLLM's ModelOpt NVFP4 linear format:
  * W: e2m1 codes packed two per byte along K ([N, K/2] uint8, element 2i in the low nibble), one E4M3 scale per 16
    elements ([N, K/16]), and a per-tensor global scale w_global: W ~ value(code) * block_scale / w_global;
  * x: quantized per 16-element block on the fly with vLLM's recipe (scaled_fp4_quant), against a static per-tensor
    global scale x_global (the checkpoint's 1 / input_scale): block_scale = e4m3(x_global * blockmax * rcp(6)),
    code = e2m1_rn(x * rcp(block_scale * rcp(x_global)));
  * y = alpha * sum(value(code_x) s_x value(code_w) s_w) + LoRA, with alpha = 1 / (x_global * w_global).
The main loop is tl.dot_scaled with e2m1 operands and E4M3 block scales; Triton 3.7 lowers it on sm_120 to
mma.sync.m16n8k64 kind::mxf4nvf4.block_scale.scale_vec::4X (tests/compile_check.py). The LoRA path (shrink of the bf16
x, expand in the GEMM epilogue, adapter-sorted tiles) is the FP8 kernels', unchanged: the adapter stays exact.
CPU interpreter: DOT_F32=True decodes both operands and uses an fp32 dot (the interpreter has no dot_scaled);
PTX=False replaces the inline-PTX conversions (RN E4M3, rcp.approx.ftz) with Triton arithmetic.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.common import shrink_choice, to_e4m3

E2M1_MAX = 6.0
SPLIT_M_MAX = 512   # the shrink splits K up to this many tokens (as fp8_lora_vllm)
_DISPATCH4: dict = {}   # shared timed dispatch tables of the NVFP4 op (as fp8_lora_vllm._DISPATCH)


class _Scratch4:
    """Buffers shared by every NVFP4 layer on a device: LoRA partials (fp32), activation codes (packed e2m1) and block
    scales (E4M3), grown to the largest request before any CUDA graph is captured."""

    def __init__(self, device):
        self.device = device
        self.hp = torch.empty(0, dtype=torch.float32, device=device)
        self.xq = torch.empty(0, dtype=torch.uint8, device=device)
        self.sxq = torch.empty(0, dtype=torch.float8_e4m3fn, device=device)

    def ensure(self, n_hp: int, n_xq: int, n_sxq: int) -> None:
        if self.hp.numel() < n_hp:
            self.hp = torch.empty(n_hp, dtype=torch.float32, device=self.device)
        if self.xq.numel() < n_xq:
            self.xq = torch.empty(n_xq, dtype=torch.uint8, device=self.device)
        if self.sxq.numel() < n_sxq:
            self.sxq = torch.empty(n_sxq, dtype=torch.float8_e4m3fn, device=self.device)


_SCRATCH4: dict = {}


def _shared_scratch4(device) -> _Scratch4:
    key = str(device)
    if key not in _SCRATCH4:
        _SCRATCH4[key] = _Scratch4(device)
    return _SCRATCH4[key]
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# ---------------------------------------------------------------------------------------------------------- helpers
@triton.jit
def _rcp(x, PTX: tl.constexpr):
    """1/x as vLLM's quantizer computes it (rcp.approx.ftz.f32); exact division off the GPU."""
    if PTX:
        return tl.inline_asm_elementwise(asm="rcp.approx.ftz.f32 $0, $1;", constraints="=r,r", args=[x],
                                         dtype=tl.float32, is_pure=True, pack=1)
    else:
        return 1.0 / x


@triton.jit
def _e2m1_code(v):
    """fp32 -> e2m1 code (bit 3 = sign), round to nearest even, saturating at 6 (= cvt.rn.satfinite.e2m1x2.f32)."""
    a = tl.abs(v)
    c = ((a > 0.25).to(tl.int32) + (a >= 0.75).to(tl.int32) + (a > 1.25).to(tl.int32) + (a >= 1.75).to(tl.int32)
         + (a > 2.5).to(tl.int32) + (a >= 3.5).to(tl.int32) + (a > 5.0).to(tl.int32))
    return (c + tl.where(v < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _e2m1_value(c):
    """e2m1 code (int32, 0..15) -> fp32; magnitudes 2..7 are the fp32 bit pattern 0x3F000000 + (mag << 22)."""
    mag = c & 7
    v = (0x3F000000 + (mag << 22)).to(tl.float32, bitcast=True)
    v = tl.where(mag == 0, 0.0, tl.where(mag == 1, 0.5, v))
    return tl.where((c & 8) != 0, -v, v)


@triton.jit
def _e4m3_rn(v, PTX: tl.constexpr):
    """Non-negative fp32 (<= 448) -> E4M3, one round-to-nearest-even step. On the GPU the PTX conversion (common.to_e4m3);
    otherwise exact arithmetic: the step of v's binade (2^-9 below the normal range) and an fp32 add of 2^23 to round
    v / step to an integer, ties to even. (The interpreter's own fp8 cast does not round to nearest.)"""
    if PTX:
        return to_e4m3(v, True)
    else:
        e = ((v.to(tl.int32, bitcast=True) >> 23) & 0xFF) - 127
        step = ((tl.maximum(e, -6) - 3 + 127) << 23).to(tl.float32, bitcast=True)
        r = ((v / step) + 8388608.0) - 8388608.0
        return (r * step).to(tl.float8e4nv)


@triton.jit
def _quant_nvfp4(x, gs, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, PTX: tl.constexpr):
    """x [BLOCK_M, BLOCK_K] fp32 -> (packed codes [BLOCK_M, BLOCK_K/2] uint8, block scales [BLOCK_M, BLOCK_K/16] e4m3),
    vLLM's scaled_fp4_quant recipe with global scale gs (a scalar multiplier)."""
    xb = tl.reshape(x, (BLOCK_M, BLOCK_K // 16, 16))
    vmax = tl.max(tl.abs(xb), axis=2)
    gsv = tl.zeros((BLOCK_M, BLOCK_K // 16), dtype=tl.float32) + gs
    six = tl.zeros((BLOCK_M, BLOCK_K // 16), dtype=tl.float32) + 6.0
    sf8 = _e4m3_rn(tl.minimum(gsv * (vmax * _rcp(six, PTX)), 448.0), PTX)
    sff = sf8.to(tl.float32)
    osc = tl.where(sff != 0.0, _rcp(sff * _rcp(gsv, PTX), PTX), 0.0)
    code = tl.reshape(_e2m1_code(xb * osc[:, :, None]), (BLOCK_M, BLOCK_K // 2, 2))
    lo, hi = tl.split(code)
    return lo | (hi << 4), sf8


@triton.jit
def _dequant_tile(p, s, ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    """packed codes [ROWS, BLOCK_K/2] + e4m3 block scales [ROWS, BLOCK_K/16] -> fp32 [ROWS, BLOCK_K] (no global)."""
    pi = p.to(tl.int32)
    v = tl.reshape(tl.join(_e2m1_value(pi & 15), _e2m1_value((pi >> 4) & 15)), (ROWS, BLOCK_K // 16, 16))
    return tl.reshape(v * s.to(tl.float32)[:, :, None], (ROWS, BLOCK_K))


# ----------------------------------------------------------------------------------------------------------- kernels
@triton.jit
def _shrink_nvfp4(X, A_PTRS, SA_PTRS, AG, HP, XQ, SXQ, GS, SORTED, SEG_IDS, SEG_N, SEG_START,
                  M, K, R, S, L1, stride_xm, stride_al, stride_am, stride_xqm, stride_sxm,
                  SPLIT_K: tl.constexpr, N_SR: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr,
                  BLOCK_K: tl.constexpr, DOT_F32: tl.constexpr = False, PTX: tl.constexpr = True,
                  A_FP8: tl.constexpr = False, A_FP4: tl.constexpr = False):
    """grid axis 2: vLLM's adapter segments 0..L (the shrink, bf16 x, as in fp8_lora_vllm._shrink_vllm), and L1 = L + 1:
    all tokens in natural order, NVFP4 x_q and block scales. Block scales are local to 16 elements, so the N_SR * SPLIT_K
    programs of a row tile each quantize their own K-slice (no full-row pass)."""
    pid = tl.program_id(0)
    seg = tl.program_id(2)
    pid_k = pid % SPLIT_K
    pid_r = (pid // SPLIT_K) % N_SR
    pid_m = pid // (SPLIT_K * N_SR)
    if seg == L1:
        if pid_m * BLOCK_M >= M:
            return
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mrow = rows < M
        gs = tl.load(GS)
        p = pid_r * SPLIT_K + pid_k
        k_slice = tl.cdiv(tl.cdiv(K, N_SR * SPLIT_K), BLOCK_K) * BLOCK_K
        q_lo = p * k_slice
        q_hi = tl.minimum(q_lo + k_slice, K)
        for k0 in range(q_lo, q_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :],
                        mask=mrow[:, None] & (offs_k[None, :] < q_hi), other=0.0).to(tl.float32)
            packed, sf = _quant_nvfp4(x, gs, BLOCK_M, BLOCK_K, PTX)
            offs_p = k0 // 2 + tl.arange(0, BLOCK_K // 2)
            tl.store(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_p[None, :], packed,
                     mask=mrow[:, None] & (offs_p[None, :] < q_hi // 2))
            offs_s = k0 // 16 + tl.arange(0, BLOCK_K // 16)
            tl.store(SXQ + rows[:, None].to(tl.int64) * stride_sxm + offs_s[None, :], sf,
                     mask=mrow[:, None] & (offs_s[None, :] < q_hi // 16))
        return
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    lora = tl.load(SEG_IDS + seg)
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
    s = c0 // R
    offs_c = (c0 - s * R) + tl.arange(0, BLOCK_SR)
    # adapter weights: bf16, FP8 (one scale per rank row) or NVFP4 (blocks of 16 along K, a global per adapter and
    # slice), dequantized in registers, as in fp8_lora_vllm._shrink_vllm (adapter_store.AdapterStore holds them)
    if A_FP8:
        a_base = tl.load(A_PTRS + s).to(tl.pointer_type(tl.float8e4nv)) + lora.to(tl.int64) * stride_al
    elif A_FP4:
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
def _gemm_nvfp4(XQ, SXQ, W, SW, ALPHA, HP, B_PTRS, SB_PTRS, BG, Y, TOK, SRT, SEG_IDS, SEG_N, SEG_START, N_SEG,
                M, N, K, R, S, END0, END1, END2, stride_xqm, stride_sxm, stride_wn, stride_swn, stride_ym,
                SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, EVEN_K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                DOT_F32: tl.constexpr = False, BASE: tl.constexpr = True,
                B_FP8: tl.constexpr = False, B_FP4: tl.constexpr = False, R_CHUNK: tl.constexpr = 64):
    """BASE=True: y = NVFP4 GEMM + LoRA. BASE=False: y += LoRA in place (y from the vendor GEMM; tiles without an
    adapter row are left untouched)."""
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
    # Tile rows as in fp8_lora_vllm._gemm_vllm: adapter-sorted order when any token has an adapter; positions past
    # the n_all tokens the metadata covers (CUDA-graph padding) run as their own rows without adapter; clamped.
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, k_end, BLOCK_K):
        offs_p = k0 // 2 + tl.arange(0, BLOCK_K // 2)
        offs_s = k0 // 16 + tl.arange(0, BLOCK_K // 16)
        if EVEN_K:
            ma = mm[:, None]
            mb = mn[:, None]
            msa = mm[:, None]
            msb = mn[:, None]
        else:
            kp = offs_p < K // 2
            ks = offs_s < K // 16
            ma = mm[:, None] & kp[None, :]
            mb = mn[:, None] & kp[None, :]
            msa = mm[:, None] & ks[None, :]
            msb = mn[:, None] & ks[None, :]
        a = tl.load(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_p[None, :], mask=ma, other=0)
        sa = tl.load(SXQ + rows[:, None].to(tl.int64) * stride_sxm + offs_s[None, :], mask=msa, other=0.0)
        b = tl.load(W + offs_n[:, None].to(tl.int64) * stride_wn + offs_p[None, :], mask=mb, other=0)
        sb = tl.load(SW + offs_n[:, None].to(tl.int64) * stride_swn + offs_s[None, :], mask=msb, other=0.0)
        if DOT_F32:
            acc = tl.dot(_dequant_tile(a, sa, BLOCK_M, BLOCK_K), tl.trans(_dequant_tile(b, sb, BLOCK_N, BLOCK_K)), acc)
        else:
            acc = tl.dot_scaled(a, sa, "e2m1", tl.trans(b), sb, "e2m1", acc)
    if BASE:
        acc = acc * tl.load(ALPHA)
    else:
        acc = tl.load(Y + rows[:, None].to(tl.int64) * stride_ym + offs_n[None, :], mask=mm[:, None] & mn[None, :],
                      other=0.0).to(tl.float32)
    if tl.max(tok, axis=0) >= 0:                                       # LoRA epilogue, as in _gemm_vllm
        n0 = pid_n * BLOCK_N
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
                if (lora >= 0) & (cnt > 0) & (st < t_lo + BLOCK_M) & (st + cnt > t_lo):
                    sel = tok == lora
                    if tl.sum(sel.to(tl.int32), axis=0) > 0:
                        h_l = tl.where(sel[:, None], hb, 0.0).to(tl.bfloat16)
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


# ---------------------------------------------------------------------------------------------------- host helpers
def nvfp4_quant_torch(x: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's reference recipe (nvfp4_emulation_utils.ref_nvfp4_quant) with exact reciprocals: x [m, k] ->
    (packed codes [m, k/2] uint8, block scales [m, k/16] float8_e4m3fn)."""
    m, k = x.shape
    xb = x.float().reshape(m, k // 16, 16)
    gs = torch.tensor(global_scale, dtype=torch.float32, device=x.device)      # fp32 throughout, as the kernels
    vmax = xb.abs().amax(-1, keepdim=True)
    sf = (gs * (vmax * (1.0 / E2M1_MAX))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sff = sf.float()
    osc = torch.where(sff != 0, 1.0 / (sff * (1.0 / gs)), torch.zeros_like(sff))
    a = (xb * osc).abs().clamp(max=E2M1_MAX)
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
            + (a >= 3.5).int() + (a > 5.0).int())
    code = (code + 8 * ((xb * osc) < 0).int()).reshape(m, k // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return packed, sf.reshape(m, k // 16)


def nvfp4_quant_batched(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 storage of a stack of matrices t [l, rows, cols] (cols % 16 == 0), one global scale per matrix (ModelOpt:
    amax -> 448 * 6), the recipe of nvfp4_quant_torch: (packed [l, rows, cols/2] uint8, block scales [l, rows, cols/16]
    float8_e4m3fn, inverse global scales [l] float32). value = code * block_scale * inv_global."""
    l, rows, cols = t.shape
    tf = t.float()
    gs = (448.0 * E2M1_MAX / tf.abs().amax(dim=(1, 2)).clamp_min(1e-12)).float()        # [l]
    xb = tf.reshape(l, rows, cols // 16, 16)
    g = gs.view(l, 1, 1, 1)
    vmax = xb.abs().amax(-1, keepdim=True)
    sf = (g * (vmax * (1.0 / E2M1_MAX))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sff = sf.float()
    osc = torch.where(sff != 0, 1.0 / (sff * (1.0 / g)), torch.zeros_like(sff))
    v = xb * osc
    a = v.abs()
    code = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int()
            + (a >= 3.5).int() + (a > 5.0).int() + 8 * (v < 0).int()).reshape(l, rows, cols // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return packed, sf.reshape(l, rows, cols // 16), (1.0 / gs).float()


def nvfp4_dequant_torch(packed: torch.Tensor, sf: torch.Tensor, global_scale: float) -> torch.Tensor:
    """(packed [m, k/2], scales [m, k/16]) -> fp32 [m, k], divided by the global scale."""
    lut = torch.tensor(E2M1_VALUES, device=packed.device)
    p = packed.to(torch.int64)
    m, kh = p.shape
    codes = torch.stack([p & 15, (p >> 4) & 15], dim=-1).reshape(m, 2 * kh)
    v = lut[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return (v.reshape(m, -1, 16) * sf.float().reshape(m, -1, 1)).reshape(m, 2 * kh) / global_scale


def global_scale_for(t: torch.Tensor) -> float:
    """ModelOpt's per-tensor global scale: the multiplier that maps amax to 448 * 6."""
    return float(448.0 * E2M1_MAX / t.abs().amax().clamp_min(1e-12))


def pick_cfg(M: int, N: int, K: int, R: int, widths) -> tuple[dict, dict]:
    """Shrink and GEMM configs by batch size, from the sm_120 tile search (job 65485500, results/nvfp4_tune.jsonl: the
    configs most often in the top 5 over both models' projections). BLOCK_N must divide every slice width; BLOCK_K >= 64
    (one m16n8k64 FP4 MMA step). At M <= 64 the base GEMM matches CUTLASS's; above, the best Triton tiles are still
    1.3-1.7x slower than CUTLASS (geomean), so large batches should use the vendor GEMM."""
    if M <= 32:
        cg = {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 4}
    elif M <= 64:
        cg = {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 4}
    elif M <= 128:
        cg = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    elif M <= 1024:
        cg = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 8, "num_stages": 3}
    else:
        cg = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3}
    while any(w % cg["BLOCK_N"] for w in widths) and cg["BLOCK_N"] > 32:
        cg["BLOCK_N"] //= 2
    cg["GROUP_M"] = 8
    cs = {"BLOCK_M": 16 if M <= 1024 else 64, "BLOCK_SR": min(64, R), "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}
    return cs, cg


class VllmFusedNVFP4LoRA:
    """Fused NVFP4 base + LoRA for one vLLM LoRA layer (the NVFP4 counterpart of fp8_lora_vllm.VllmFusedFP8LoRA).

    w_q [N, K/2] uint8 (packed e2m1), w_sf [N, K/16] float8_e4m3fn (linear layout), w_global and x_global the global
    scale multipliers (W ~ value * sf / w_global; x quantized against x_global); a_stacked / b_stacked: vLLM's
    per-slice buffers [L, 1, R, K] / [L, 1, N_s, R] (bf16). max_m: the engine's max tokens per forward.
    """

    def __init__(self, w_q, w_sf, w_global, x_global, a_stacked, b_stacked, max_m, *, split_k=None, lora_fmt="bf16"):
        self.w_q, self.w_sf = w_q, w_sf
        self.N, kh = w_q.shape
        self.K = 2 * kh
        assert w_sf.shape == (self.N, self.K // 16) and self.K % 64 == 0
        dev = w_q.device
        self.x_global = torch.tensor([x_global], dtype=torch.float32, device=dev)
        self.alpha = torch.tensor([1.0 / (x_global * w_global)], dtype=torch.float32, device=dev)
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
        self.a_stacked, self.b_stacked = a_stacked, b_stacked
        from lowbit_lora.kernels.adapter_store import AdapterStore   # (lazy: adapter_store imports this module)
        self.store = AdapterStore(a_stacked, b_stacked, lora_fmt)    # bf16 = vLLM's own buffers; fp8 / nvfp4 copies
        self.lora_fmt = lora_fmt
        self.requantize = self.store.requantize
        self.stride_al, self.stride_am = a_stacked[0].stride(0), a_stacked[0].stride(2)
        self.split_k = split_k or max(1, min(16, self.K // 512))
        self.cc_major = torch.cuda.get_device_capability(dev)[0] if dev.type == "cuda" else 0
        self.max_m = max_m
        # scratch shared by every NVFP4 layer of the model, sized for the largest (layers run one after another on
        # one stream, also inside a CUDA graph): per-layer buffers cost ~12 GB of KV cache on an 8B model (job 65490036)
        self.scratch = _shared_scratch4(dev)
        self.scratch.ensure(max(self.split_k * self.S * min(SPLIT_M_MAX, max_m), self.S * max_m) * self.R,
                            max_m * (self.K // 2), max_m * (self.K // 16))
        self.n_seg = torch.full((1,), self.L + 1, dtype=torch.int32, device=dev)
        self.dot_f32 = False                    # CPU interpreter: decode operands and use an fp32 dot
        self.base_fn = None                     # the vendor NVFP4 GEMM (x -> y) for the hybrid path, if set
        # bucket of M -> "fused" | "hybrid" | "hybrids" (tune_dispatch; empty: fused up to hybrid_m, hybrid above).
        # "hybrids": the vendor GEMM, then vLLM's own LoRA kernels on vLLM's bf16 adapters (bf16 adapters only)
        self.dispatch = {}
        self.r_chunk = None                     # the GEMM epilogue's rank chunk: None = auto
        self._no_lora_cpu = torch.zeros(1, dtype=torch.bool)
        self._n_active_cpu = torch.tensor([self.L + 1], dtype=torch.int32)

    def release(self, a_stage, b_stage) -> None:
        """Serve from the store's low-precision copies alone (AdapterStore.release): one bf16 staging slot."""
        self.store.release(a_stage, b_stage)
        self.a_stacked, self.b_stacked = a_stage, b_stage

    @property
    def hp(self):
        return self.scratch.hp

    @property
    def xq(self):
        return self.scratch.xq[: self.max_m * (self.K // 2)].view(self.max_m, self.K // 2)

    @property
    def sxq(self):
        return self.scratch.sxq[: self.max_m * (self.K // 16)].view(self.max_m, self.K // 16)

    def _run_stock(self, x, meta_args):
        """The vendor GEMM, then vLLM's own LoRA shrink and expand (Punica kernels) on vLLM's bf16 adapter buffers."""
        from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
        tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
        M = x.shape[0]
        y = self.base_fn(x)
        hb = self.hp[: self.S * M * self.R].view(self.S, M, self.R)
        hb.zero_()
        m7 = (tok, srt, seg_n, seg_start, seg_ids, self._no_lora_cpu, self._n_active_cpu)
        lora_shrink(x, list(self.a_stacked), hb, *m7, 1.0)
        lora_expand(hb, list(self.b_stacked), y, *m7, offset_start=0, add_inputs=True)
        return y

    def _dummy_meta(self, M: int, lora: int):
        dev = self.w_q.device
        tok = torch.full((M,), lora, dtype=torch.int32, device=dev)
        srt = torch.arange(M, dtype=torch.int32, device=dev)
        seg_ids = torch.full((self.L + 1,), -1, dtype=torch.int32, device=dev)
        seg_n = torch.zeros(self.L + 1, dtype=torch.int32, device=dev)
        seg_start = torch.zeros(self.L + 2, dtype=torch.int32, device=dev)
        if lora >= 0:
            seg_ids[0], seg_n[0] = lora, M
            seg_start[1] = M
        return tok, srt, seg_n, seg_start, seg_ids

    def tune_dispatch(self) -> dict:
        """Per bucket of M, time (CUDA-graph replays) our fused NVFP4 kernels (M <= 128: above, CUTLASS's FP4 GEMM is
        faster than any of our tilings), the hybrid (vendor GEMM + our LoRA kernels) and, for bf16 adapters, the vendor
        GEMM + vLLM's own LoRA kernels, and keep the fastest; layers of one shape share the table."""
        if self.base_fn is None:
            return self.dispatch
        from lowbit_lora.kernels.fp8_lora_vllm import _BUCKETS, _graph_us   # (lazy: that module imports this one)
        key = ("nvfp4", torch.cuda.get_device_name(self.w_q.device), self.N, self.K, self.R, tuple(self.widths),
               self.lora_fmt)
        if key not in _DISPATCH4:
            table, prev = {}, None
            for b in _BUCKETS:
                M = min(b, self.max_m)
                if M < 1 or M == prev:
                    break
                prev = M
                x = torch.randn(M, self.K, device=self.w_q.device).to(torch.bfloat16)
                meta = self._dummy_meta(M, 0)
                t = {"hybrid": _graph_us(lambda: self(x, meta, base_fn=self.base_fn, hybrid_m=0, use_dispatch=False))}
                if M <= 128:
                    t["fused"] = _graph_us(lambda: self(x, meta, hybrid_m=1 << 30, use_dispatch=False))
                if self.lora_fmt == "bf16":
                    t["hybrids"] = _graph_us(lambda: self._run_stock(x, meta))
                table[b] = min(t, key=t.get)
            _DISPATCH4[key] = table
        self.dispatch = _DISPATCH4[key]
        return self.dispatch

    def warmup(self) -> None:
        """Time the dispatch table (if a vendor GEMM is set), then compile every variant pick_cfg can select (divisible
        and non-divisible M per bucket), with and without an adapter, on every path, before serving (as
        fp8_lora_vllm.VllmFusedFP8LoRA.warmup)."""
        self.tune_dispatch()
        dev = self.w_q.device
        sizes = [m for m in (1, 2, 16, 17, 48, 49, 96, 97, 192, 193, 512, 513, 2048, 2049, self.max_m, self.max_m - 1)
                 if 1 <= m <= self.max_m]
        for M in sizes:
            x = torch.zeros(M, self.K, dtype=torch.bfloat16, device=dev)
            for lora in (0, -1):
                tok, srt, seg_n, seg_start, seg_ids = self._dummy_meta(M, lora)   # (L + 2 start offsets, as vLLM's)
                self(x, (tok, srt, seg_n, seg_start, seg_ids))
                if self.base_fn is not None:   # every path the dispatch may pick
                    self(x, (tok, srt, seg_n, seg_start, seg_ids), hybrid_m=0, use_dispatch=False)
                    self(x, (tok, srt, seg_n, seg_start, seg_ids), hybrid_m=1 << 30, use_dispatch=False)
                    if self.lora_fmt == "bf16":
                        self._run_stock(x, (tok, srt, seg_n, seg_start, seg_ids))
        torch.cuda.synchronize()

    def __call__(self, x, meta_args, out=None, cfg_s=None, cfg_g=None, base_fn=None, hybrid_m=64, split=None,
                 use_dispatch=True):
        """base_fn (the vendor NVFP4 GEMM, x -> y): for M > hybrid_m, y = base_fn(x) and the LoRA is added in place
        (our shrink without the x_q pass, then the epilogue kernel with BASE=False); else our fused kernels.
        split: override shrink_choice's split-K (tile search; the scratch holds split <= split_k up to SPLIT_M_MAX)."""
        tok, srt, seg_n, seg_start, seg_ids = meta_args[:5]
        M = x.shape[0]
        if use_dispatch and base_fn is None and self.dispatch:   # the plugin's call: the timed table decides
            from lowbit_lora.kernels.fp8_lora_vllm import _bucket
            d = self.dispatch.get(_bucket(M))
            if d == "hybrids":
                return self._run_stock(x, meta_args)
            hybrid_m = 1 << 30 if d == "fused" else 0
        base_fn = base_fn if base_fn is not None else self.base_fn
        hybrid = base_fn is not None and M > hybrid_m
        cs_t, split_t = shrink_choice(M, self.R, self.split_k, self.cc_major)   # tuned per architecture
        if split is None:
            split = split_t
            while split > 1 and split * self.S * M * self.R > self.hp.numel():
                split //= 2                                              # what the shared scratch holds
        else:   # an explicit split (tile search): any that the shared scratch holds, also above SPLIT_M_MAX tokens
            assert split == 1 or split * self.S * M * self.R <= self.hp.numel()
        ds, dg = pick_cfg(M, self.N, self.K, self.R, self.widths)
        cs, cg = cfg_s or cs_t, cfg_g or dg
        assert self.R % cs["BLOCK_SR"] == 0 and all(w % cg["BLOCK_N"] == 0 for w in self.widths)
        n_sr = self.S * self.R // cs["BLOCK_SR"]
        ptx = not self.dot_f32
        grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, self.L + (1 if hybrid else 2))   # L1 = x_q pass
        _shrink_nvfp4[grid_s](
            x, self.store.a_ptrs, self.store.sa_ptrs, self.store.ag, self.hp, self.xq, self.sxq, self.x_global, srt,
            seg_ids, seg_n, seg_start,
            M, self.K, self.R, self.S, self.L + 1, x.stride(0), self.stride_al, self.stride_am, self.xq.stride(0),
            self.sxq.stride(0), SPLIT_K=split, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
            BLOCK_K=cs["BLOCK_K"], DOT_F32=self.dot_f32, PTX=ptx, A_FP8=self.lora_fmt == "fp8",
            A_FP4=self.lora_fmt == "nvfp4", num_warps=cs["num_warps"], num_stages=cs["num_stages"])
        if hybrid:
            y = base_fn(x)
        else:
            y = out if out is not None else torch.empty(M, self.N, dtype=torch.bfloat16, device=x.device)
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        br = max(16, triton.next_power_of_2(self.R))
        # The epilogue's rank chunk: 64 for NVFP4-stored B and for the expand-add (BASE=False), whose whole-rank tiles
        # spill at rank 128 (as in fp8_lora_vllm); the whole rank otherwise, where one
        # dot is faster (bf16/FP8 fused: 1.15-1.25x slower in chunks). r_chunk forces a chunk (tests).
        rck = self.r_chunk or (64 if (self.lora_fmt == "nvfp4" or hybrid) else br)
        _gemm_nvfp4[grid_g](
            self.xq, self.sxq, self.w_q, self.w_sf, self.alpha, self.hp, self.store.b_ptrs, self.store.sb_ptrs,
            self.store.bg, y, tok, srt, seg_ids, seg_n, seg_start, self.n_seg, M, self.N, self.K, self.R, self.S, self.end0, self.end1, self.end2,
            self.xq.stride(0), self.sxq.stride(0), self.w_q.stride(0), self.w_sf.stride(0), y.stride(0),
            SPLIT_K1=split, BLOCK_R=br, EVEN_K=self.K % cg["BLOCK_K"] == 0,
            BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
            DOT_F32=self.dot_f32, BASE=not hybrid, B_FP8=self.lora_fmt == "fp8", B_FP4=self.lora_fmt == "nvfp4",
            R_CHUNK=rck, num_warps=cg["num_warps"], num_stages=cg["num_stages"])
        return y


def reference(x, tok, w_q, w_sf, w_global, x_global, A, B, widths, xq=None, xsf=None):
    """fp32 reference: NVFP4-quantized x (vLLM's recipe, exact reciprocals; or the given codes xq/xsf) times the
    dequantized W, plus the exact LoRA path h = x A_l^T (bf16 x, fp32), expand h_s B_s^T per slice;
    A [L, S*R, K], B [L, N, R]; tok [M] (-1 = none)."""
    if xq is None:
        xq, xsf = nvfp4_quant_torch(x, x_global)
    y = nvfp4_dequant_torch(xq, xsf, x_global) @ nvfp4_dequant_torch(w_q, w_sf, w_global).t()
    R = B.shape[2]
    offs = [sum(widths[:i]) for i in range(len(widths))]
    for i in range(x.shape[0]):
        lo = int(tok[i])
        if lo < 0:
            continue
        h = A[lo].float() @ x[i].float()                                  # [S*R]
        for s, (o, w) in enumerate(zip(offs, widths)):
            y[i, o:o + w] += B[lo, o:o + w].float() @ h[s * R:(s + 1) * R].to(torch.bfloat16).float()
    return y
