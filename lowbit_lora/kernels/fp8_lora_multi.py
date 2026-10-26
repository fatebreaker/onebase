"""Fused FP8 GEMM + multi-adapter LoRA (Phase 1; the single-adapter Phase-0 prototype is fp8_lora.py).

    y[m] = s_x[m] * s_w * (x_q[m] @ W_q^T)  +  h[m]_s @ B[l(m)]_s^T,     h[m] = scale * x[m] @ A[l(m)]^T

for a batch whose tokens use different adapters l(m) (slot indices into stacked adapter weights; -1 = no adapter).

  _shrink_multi   grid (M-tiles x SR-blocks x split-K, 1, segments): one program per (adapter segment, tile) as in
                  Punica's SGMV, gathering the segment's token rows (tokens sorted by adapter). It writes split-K
                  partials of h by token ([SPLIT_K, S, M, R], token order) and per-token |x| maxima; the segment of
                  tokens without an adapter only computes maxima. In prefill mode (SPLIT_K = 1) it also writes x_q.
  _gemm_multi     the FP8 GEMM over tiles of the NATURAL token order, so W is read once per tile whatever the number
                  of adapters; the epilogue loops over the batch's active adapters, skips those absent from the tile,
                  and adds h_rows(l) @ B[l]_s^T for the rows of adapter l. Adapter bytes are read once per output tile
                  per adapter present, the inherent minimum.
Adapters are stacked [L, S*r, K] (A) and [L, N, r] (B), bf16 or FP8 with per-row float32 scales [L, S*r] / [L, N].
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.common import to_e4m3

FP8_MAX = 448.0
_FP8_MAX = tl.constexpr(448.0)   # the same constant, usable inside @triton.jit kernels
_MIN_SCALE = tl.constexpr(1.0 / (448.0 * 512.0))   # vLLM's floor on a per-token scale


def segments(token_lora: torch.Tensor, max_loras: int) -> dict:
    """Punica-style metadata: tokens sorted by adapter slot; segment s covers sorted[start[s] : start[s] + n[s]] and
    uses slot ids[s] (-1 = no adapter). Fixed shapes [max_loras + 1] so that a call can be graph-captured."""
    dev = token_lora.device
    order = torch.argsort(token_lora, stable=True).to(torch.int32)
    ids_all = torch.arange(-1, max_loras, device=dev, dtype=torch.int32)                 # -1, 0, 1, ..., L-1
    counts = torch.stack([(token_lora == i).sum() for i in ids_all]).to(torch.int32)
    start = torch.zeros(max_loras + 1, dtype=torch.int32, device=dev)
    start[1:] = torch.cumsum(counts, 0)[:-1].to(torch.int32)
    active = ids_all[(counts > 0) & (ids_all >= 0)]
    act = torch.full((max_loras,), -1, dtype=torch.int32, device=dev)
    act[: active.numel()] = active
    return {"sorted": order, "seg_ids": ids_all, "seg_n": counts, "seg_start": start, "active": act,
            "n_active": int(active.numel())}


@triton.jit
def _shrink_multi(X, A, SA, HP, PMAX, XQ, SX, SORTED, SEG_IDS, SEG_N, SEG_START,
                  M, K, SR, R, S, scale, stride_xm, stride_al, stride_am, stride_sal, stride_xqm,
                  SPLIT_K: tl.constexpr, WRITE_XQ: tl.constexpr, N_SR: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr, BLOCK_K: tl.constexpr,
                  A_FP8: tl.constexpr = False, DOT_F32: tl.constexpr = False, RN: tl.constexpr = True):
    pid = tl.program_id(0)
    seg = tl.program_id(2)
    pid_k = pid % SPLIT_K
    pid_r = (pid // SPLIT_K) % N_SR
    pid_m = pid // (SPLIT_K * N_SR)
    lora = tl.load(SEG_IDS + seg)
    seg_n = tl.load(SEG_N + seg)
    if pid_m * BLOCK_M >= seg_n:
        return
    if lora < 0 and pid_r > 0 and not WRITE_XQ:      # tokens without an adapter: maxima only (column block 0)
        return
    seg_start = tl.load(SEG_START + seg)
    offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mrow = offs < seg_n
    rows = tl.load(SORTED + seg_start + offs, mask=mrow, other=0)                      # token indices
    offs_c = pid_r * BLOCK_SR + tl.arange(0, BLOCK_SR)
    mc = offs_c < SR
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_SR), dtype=tl.float32)
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    a_base = A + tl.maximum(lora, 0).to(tl.int64) * stride_al
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < k_hi
        x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=mrow[:, None] & mk[None, :],
                    other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
        if lora >= 0:
            a = tl.load(a_base + offs_c[:, None] * stride_am + offs_k[None, :], mask=mc[:, None] & mk[None, :],
                        other=0.0)
            if A_FP8:
                a = a.to(tl.bfloat16)
            if DOT_F32:   # CPU interpreter only: its bf16/fp8 tl.dot is wrong
                acc = tl.dot(x.to(tl.float32), tl.trans(a.to(tl.float32)), acc)
            else:
                acc = tl.dot(x, tl.trans(a), acc)
    if lora >= 0:   # h partial by TOKEN, layout [SPLIT_K, S, M, R]
        if A_FP8:
            acc = acc * tl.load(SA + tl.maximum(lora, 0) * stride_sal + offs_c, mask=mc, other=0.0)[None, :]
        s_idx = offs_c // R
        r_idx = offs_c % R
        hp = HP + pid_k * (S * M * R) + s_idx[None, :] * (M * R) + rows[:, None] * R + r_idx[None, :]
        tl.store(hp, acc * scale, mask=mrow[:, None] & mc[None, :])
    if pid_r == 0:
        tl.store(PMAX + pid_k * M + rows, amax, mask=mrow)
    if WRITE_XQ:   # SPLIT_K == 1: full maxima known; the column programs share the x_q pass over K-slices
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
        if pid_r == 0:
            tl.store(SX + rows, sx, mask=mrow)
        k_slice = tl.cdiv(tl.cdiv(K, N_SR), BLOCK_K) * BLOCK_K
        q_lo = pid_r * k_slice
        q_hi = tl.minimum(q_lo + k_slice, K)
        for k0 in range(q_lo, q_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            msk = mrow[:, None] & (offs_k[None, :] < q_hi)
            x = tl.load(X + rows[:, None].to(tl.int64) * stride_xm + offs_k[None, :], mask=msk, other=0.0)
            q = to_e4m3(tl.clamp(tl.math.div_rn(x.to(tl.float32), sx[:, None]), -_FP8_MAX, _FP8_MAX), RN)
            tl.store(XQ + rows[:, None].to(tl.int64) * stride_xqm + offs_k[None, :], q, mask=msk)


@triton.jit
def _gemm_multi(X, XQ, SX, PMAX, W, SW, HP, B, SB, Y, TOK, ACTIVE, N_ACTIVE,
                M, N, K, R, S, END0, END1,
                stride_xm, stride_xqm, stride_wn, stride_bl, stride_bn, stride_sbl, stride_ym,
                X_FP8: tl.constexpr, SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, EVEN_K: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                B_FP8: tl.constexpr = False, DOT_F32: tl.constexpr = False, RN: tl.constexpr = True, PRECISE: tl.constexpr = True):
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
    if X_FP8:
        sx = tl.load(SX + offs_m, mask=mm, other=1.0)
    else:
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for j in tl.static_range(SPLIT_K1):
            amax = tl.maximum(amax, tl.load(PMAX + j * M + offs_m, mask=mm, other=0.0))
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        if EVEN_K:
            ma = mm[:, None]
            mb = mn[None, :]
        else:
            ma = mm[:, None] & (offs_k[None, :] < K)
            mb = mn[None, :] & (offs_k[:, None] < K)
        if X_FP8:
            a = tl.load(XQ + offs_m[:, None] * stride_xqm + offs_k[None, :], mask=ma, other=0.0)
        else:
            xb = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=ma, other=0.0).to(tl.float32)
            a = to_e4m3(tl.clamp(tl.math.div_rn(xb, sx[:, None]), -_FP8_MAX, _FP8_MAX), RN)
        b = tl.load(W + offs_n[None, :] * stride_wn + offs_k[:, None], mask=mb, other=0.0)
        if DOT_F32:
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
        elif PRECISE:   # FP8 MMA accumulates with reduced precision: promote to fp32 after every K-block, as CUTLASS does
            acc += tl.dot(a, b)
        else:
            acc = tl.dot(a, b, acc)
    acc = acc * (sx[:, None] * tl.load(SW))
    n_active = tl.load(N_ACTIVE)
    if n_active > 0:
        tok = tl.load(TOK + offs_m, mask=mm, other=-1)
        n0 = pid_n * BLOCK_N
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32)   # a tile never straddles two slices
        offs_r = tl.arange(0, BLOCK_R)
        mr = offs_r < R
        h = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        hm = mm & (tok >= 0)
        for j in tl.static_range(SPLIT_K1):
            h += tl.load(HP + j * (S * M * R) + s * (M * R) + offs_m[:, None] * R + offs_r[None, :],
                         mask=hm[:, None] & mr[None, :], other=0.0)
        hb = h.to(tl.bfloat16)
        for i in range(0, n_active):
            lora = tl.load(ACTIVE + i)
            sel = tok == lora
            if tl.sum(sel.to(tl.int32), axis=0) > 0:
                h_l = tl.where(sel[:, None], hb, 0.0).to(tl.bfloat16)
                bt = tl.load(B + lora.to(tl.int64) * stride_bl + offs_n[None, :] * stride_bn + offs_r[:, None],
                             mask=mn[None, :] & mr[:, None], other=0.0)
                if B_FP8:
                    bt = bt.to(tl.bfloat16)
                if DOT_F32:
                    lo = tl.dot(h_l.to(tl.float32), bt.to(tl.float32))
                else:
                    lo = tl.dot(h_l, bt)
                if B_FP8:
                    lo = lo * tl.load(SB + lora * stride_sbl + offs_n, mask=mn, other=0.0)[None, :]
                acc += lo
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


class MultiFusedFP8LoRA:
    """One fused projection serving up to max_loras adapters at once, with static buffers (graph-capturable).

    w_q [N, K] float8_e4m3fn, s_w [1] float32; a [L, S*r, K], b [L, N, r] (bf16, or FP8 with a_scale [L, S*r] and
    b_scale [L, N] float32); slices: output size per slice; scale = alpha / r (one value for all adapters).
    """

    def __init__(self, w_q, s_w, a, b, slices, scale, max_m, *, split_k=None, a_scale=None, b_scale=None):
        self.w_q, self.s_w, self.a, self.b = w_q, s_w, a, b
        self.N, self.K = w_q.shape
        self.L = a.shape[0]
        self.S = len(slices)
        self.R = b.shape[2]
        assert a.shape[1:] == (self.S * self.R, self.K) and b.shape[1:] == (self.N, self.R) and sum(slices) == self.N
        ends = [sum(slices[:i + 1]) for i in range(self.S)]
        self.end0 = ends[0] if self.S >= 2 else self.N
        self.end1 = ends[1] if self.S >= 3 else self.N
        self.slices, self.scale = tuple(slices), float(scale)
        self.a_fp8, self.b_fp8 = a_scale is not None, b_scale is not None
        self.a_scale = a_scale if a_scale is not None else torch.zeros(1, 1, device=w_q.device)
        self.b_scale = b_scale if b_scale is not None else torch.zeros(1, 1, device=w_q.device)
        self.split_k = split_k or max(1, min(16, self.K // 512))
        dev = w_q.device
        self.hp = torch.empty(self.split_k * self.S * max_m * self.R, dtype=torch.float32, device=dev)
        self.pmax = torch.empty(self.split_k * max_m, dtype=torch.float32, device=dev)
        self.xq = torch.empty(max_m, self.K, dtype=torch.float8_e4m3fn, device=dev)
        self.sx = torch.empty(max_m, dtype=torch.float32, device=dev)
        self.y = torch.empty(max_m, self.N, dtype=torch.bfloat16, device=dev)
        self.n_active = torch.zeros(1, dtype=torch.int32, device=dev)
        self.dot_f32 = False          # True only under TRITON_INTERPRET=1 (tests on CPU)

    def __call__(self, x, token_lora, meta, mode, cfg_s=None, cfg_g=None):
        M = x.shape[0]
        prefill = mode == "prefill"
        split = 1 if prefill else self.split_k
        cs = cfg_s or {"BLOCK_M": 16, "BLOCK_SR": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}
        cg = cfg_g or ({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}
                       if M <= 64 else {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,
                                        "num_warps": 8, "num_stages": 3})
        SR = self.S * self.R
        n_sr = triton.cdiv(SR, cs["BLOCK_SR"])
        self.n_active.fill_(meta["n_active"])
        grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, self.L + 1)
        _shrink_multi[grid_s](
            x, self.a, self.a_scale, self.hp, self.pmax, self.xq, self.sx, meta["sorted"], meta["seg_ids"],
            meta["seg_n"], meta["seg_start"], M, self.K, SR, self.R, self.S, self.scale, x.stride(0),
            self.a.stride(0), self.a.stride(1), self.a_scale.stride(0), self.xq.stride(0),
            SPLIT_K=split, WRITE_XQ=prefill, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
            BLOCK_K=cs["BLOCK_K"], A_FP8=self.a_fp8, DOT_F32=self.dot_f32, RN=not self.dot_f32, num_warps=cs["num_warps"],
            num_stages=cs["num_stages"])
        assert all(sz % cg["BLOCK_N"] == 0 for sz in self.slices), (self.slices, cg["BLOCK_N"])
        y = self.y[:M]
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        _gemm_multi[grid_g](
            x, self.xq, self.sx, self.pmax, self.w_q, self.s_w, self.hp, self.b, self.b_scale, y, token_lora,
            meta["active"], self.n_active, M, self.N, self.K, self.R, self.S, self.end0, self.end1,
            x.stride(0), self.xq.stride(0), self.w_q.stride(0), self.b.stride(0), self.b.stride(1),
            self.b_scale.stride(0), y.stride(0),
            X_FP8=prefill, SPLIT_K1=split, BLOCK_R=max(16, triton.next_power_of_2(self.R)),
            EVEN_K=self.K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"],
            GROUP_M=cg["GROUP_M"], PRECISE=os.environ.get("LOWBIT_PRECISE", "1") != "0", B_FP8=self.b_fp8, DOT_F32=self.dot_f32, RN=not self.dot_f32, num_warps=cg["num_warps"],
            num_stages=cg["num_stages"])
        return y


    def tune(self, x, token_lora, meta, mode, rep=25) -> tuple[dict, dict]:
        """Fastest shrink config, then fastest GEMM config, for this batch (eager timing; never inside a graph)."""
        from triton.testing import do_bench
        M = x.shape[0]
        cap = max(16, triton.next_power_of_2(M))
        best_s, t_best = None, float("inf")
        for bm in (16, 32, 64):
            for bsr in (32, 64, 128):
                for bk in (64, 128):
                    if bm > cap * 2 or bsr > max(16, triton.next_power_of_2(self.S * self.R)):
                        continue
                    cs = {"BLOCK_M": bm, "BLOCK_SR": bsr, "BLOCK_K": bk, "num_warps": 4, "num_stages": 3}
                    try:
                        t = do_bench(lambda: self(x, token_lora, meta, mode, cfg_s=cs), rep=rep)
                    except Exception:   # noqa: BLE001 -- e.g. out of shared memory: skip
                        continue
                    if t < t_best:
                        t_best, best_s = t, cs
        best_g, t_best = None, float("inf")
        for bm, bn, bk, nw, ns in [(16, 64, 128, 4, 4), (16, 128, 128, 4, 4), (32, 64, 128, 4, 4), (32, 128, 128, 4, 4),
                                   (64, 64, 128, 4, 4), (64, 128, 128, 4, 4), (64, 128, 128, 8, 4), (128, 128, 128, 8, 3),
                                   (128, 128, 64, 4, 4), (128, 256, 64, 8, 3)]:
            if bm > cap * 2 or any(sz % bn for sz in self.slices):
                continue
            cg = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8, "num_warps": nw, "num_stages": ns}
            try:
                t = do_bench(lambda: self(x, token_lora, meta, mode, cfg_s=best_s, cfg_g=cg), rep=rep)
            except Exception:   # noqa: BLE001
                continue
            if t < t_best:
                t_best, best_g = t, cg
        return best_s, best_g


def reference_multi(x, token_lora, w_q, s_w, a, b, slices, scale, a_scale=None, b_scale=None, xq=None, sx=None):
    """fp32 reference: the base GEMM on FP8 x (given x_q / s_x, or quantized here) plus each token's own adapter."""
    xf = x.float()
    if xq is None:
        sx = (xf.abs().amax(dim=1) / FP8_MAX).clamp_min(1.0 / (FP8_MAX * 512.0))
        xq = (xf / sx[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    y = (xq.float() * sx[:, None]) @ (w_q.float() * s_w.float()).t()
    R = b.shape[2]
    for m in range(x.shape[0]):
        lo = int(token_lora[m])
        if lo < 0:
            continue
        am = a[lo].float() * (a_scale[lo][:, None] if a_scale is not None else 1.0)
        bm = b[lo].float() * (b_scale[lo][:, None] if b_scale is not None else 1.0)
        h = (xf[m] @ am.t() * scale).to(torch.bfloat16).float()
        off = 0
        for s_, n in enumerate(slices):
            y[m, off:off + n] += bm[off:off + n] @ h[s_ * R:(s_ + 1) * R]
            off += n
    return y
