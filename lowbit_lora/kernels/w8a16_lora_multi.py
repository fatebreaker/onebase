"""Fused weight-only FP8 (W8A16) GEMM + multi-adapter LoRA for the rollout engine (RL-MATCHED-KERNELS, docs/PLAN_MLSYS_V2.md):
the engine-side twin of the trainer's W8A16 kernel (fp8_train_kernels._gemm_w8a16_kernel).

    y[m] = s_w * (x[m] @ widen(W_q)^T)  +  h[m]_s @ B[l(m)]_s^T,     h[m] = scale * x[m] @ A[l(m)]^T

The activations stay bf16 -- no activation rounding, the part of a W8A8 engine's arithmetic that no trainer can reproduce
(MISMATCH-BENCH: the trainer's W8A8 forward raised the mismatch from 0.018 to 0.023) -- while the e4m3 weight codes are read
as bytes and widened to bf16 in registers (exact), the MMA accumulates in fp32 and the per-tensor scale is applied once at
the end, as the trainer's W8A16 kernel does: engine and trainer compute the same linear layers from the same codes. Memory
and weight bytes stay FP8; decode, memory-bound, reads the codes at the FP8 byte rate; prefill runs on bf16 tensor cores.
The shrink is fp8_lora_multi._shrink_multi without its quantization pass; the epilogue loops over the batch's active
adapters as fp8_lora_multi._gemm_multi does (each adapter's expand over its rows; adapters bf16 or FP8-stored).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from lowbit_lora.kernels.fp8_lora_multi import FP8_MAX, _shrink_multi, segments  # noqa: F401  (segments: re-exported)


@triton.jit
def _gemm_multi_w16(X, W, SW, HP, B, SB, Y, TOK, ACTIVE, N_ACTIVE,
                    M, N, K, R, S, END0, END1,
                    stride_xm, stride_wn, stride_bl, stride_bn, stride_sbl, stride_ym,
                    SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr, EVEN_K: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                    B_FP8: tl.constexpr = False, DOT_F32: tl.constexpr = False):
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
        if EVEN_K:
            ma = mm[:, None]
            mb = mn[None, :]
        else:
            ma = mm[:, None] & (offs_k[None, :] < K)
            mb = mn[None, :] & (offs_k[:, None] < K)
        a = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=ma, other=0.0)          # bf16 activations
        b = tl.load(W + offs_n[None, :] * stride_wn + offs_k[:, None], mask=mb, other=0.0).to(tl.bfloat16)   # codes, exact
        if DOT_F32:   # CPU interpreter only: its bf16 tl.dot is wrong
            acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc)
        else:
            acc = tl.dot(a, b, acc)
    acc = acc * tl.load(SW)
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


class MultiFusedW8A16LoRA:
    """One fused weight-only FP8 projection serving up to max_loras adapters at once, with static buffers
    (graph-capturable). w_q [N, K] float8_e4m3fn, s_w [1] float32; a [L, S*r, K], b [L, N, r] (bf16, or FP8 with
    a_scale [L, S*r] and b_scale [L, N] float32); slices: output size per slice; scale = alpha / r."""

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
        self.pmax = torch.empty(self.split_k * max_m, dtype=torch.float32, device=dev)   # written by the shared shrink
        self.y = torch.empty(max_m, self.N, dtype=torch.bfloat16, device=dev)
        self.n_active = torch.zeros(1, dtype=torch.int32, device=dev)
        self.dot_f32 = False          # True only under TRITON_INTERPRET=1 (tests on CPU)

    def __call__(self, x, token_lora, meta, mode, cfg_s=None, cfg_g=None):
        M = x.shape[0]
        split = self.split_k   # no activation quantization, so no full-row maxima to wait for: split K in every mode
                               # (the W8A8 kernel's prefill pass needed them; here a long-K prefill would under-fill the GPU)
        cs = cfg_s or {"BLOCK_M": 16, "BLOCK_SR": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}
        cg = cfg_g or ({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 4, "num_stages": 4}
                       if M <= 64 else {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8,
                                        "num_warps": 8, "num_stages": 3})
        SR = self.S * self.R
        n_sr = triton.cdiv(SR, cs["BLOCK_SR"])
        self.n_active.fill_(meta["n_active"])
        if meta["n_active"] > 0:      # the shrink, without its quantization pass (WRITE_XQ False; its maxima unused)
            grid_s = (triton.cdiv(M, cs["BLOCK_M"]) * n_sr * split, 1, self.L + 1)
            _shrink_multi[grid_s](
                x, self.a, self.a_scale, self.hp, self.pmax, self.y, self.y, meta["sorted"], meta["seg_ids"],
                meta["seg_n"], meta["seg_start"], M, self.K, SR, self.R, self.S, self.scale, x.stride(0),
                self.a.stride(0), self.a.stride(1), self.a_scale.stride(0), 0,
                SPLIT_K=split, WRITE_XQ=False, N_SR=n_sr, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"],
                BLOCK_K=cs["BLOCK_K"], A_FP8=self.a_fp8, DOT_F32=self.dot_f32, RN=not self.dot_f32,
                num_warps=cs["num_warps"], num_stages=cs["num_stages"])
        assert all(sz % cg["BLOCK_N"] == 0 for sz in self.slices), (self.slices, cg["BLOCK_N"])
        y = self.y[:M]
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        _gemm_multi_w16[grid_g](
            x, self.w_q, self.s_w, self.hp, self.b, self.b_scale, y, token_lora, meta["active"], self.n_active,
            M, self.N, self.K, self.R, self.S, self.end0, self.end1,
            x.stride(0), self.w_q.stride(0), self.b.stride(0), self.b.stride(1), self.b_scale.stride(0), y.stride(0),
            SPLIT_K1=split, BLOCK_R=max(16, triton.next_power_of_2(self.R)), EVEN_K=self.K % cg["BLOCK_K"] == 0,
            BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"],
            B_FP8=self.b_fp8, DOT_F32=self.dot_f32, num_warps=cg["num_warps"], num_stages=cg["num_stages"])
        return y

    def tune(self, x, token_lora, meta, mode, rep=25) -> tuple[dict, dict]:
        """Fastest shrink config, then fastest GEMM config, for this batch (eager timing; never inside a graph)."""
        from triton.testing import do_bench
        M = x.shape[0]
        cap = max(16, triton.next_power_of_2(M))
        best_s, t_best = None, float("inf")
        if meta["n_active"] > 0:
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
        for bm, bn, bk, nw, ns in [(16, 32, 128, 4, 4), (16, 64, 128, 4, 4), (16, 128, 128, 4, 4), (32, 64, 128, 4, 4),
                                   (32, 128, 128, 4, 4), (64, 64, 128, 4, 4), (64, 128, 64, 4, 4), (64, 128, 128, 8, 4),
                                   (128, 64, 64, 4, 4), (128, 128, 64, 8, 3), (128, 256, 64, 8, 3), (128, 128, 32, 4, 4),
                                   (64, 256, 64, 8, 3)]:
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


def reference_multi_w16(x, token_lora, w_q, s_w, a, b, slices, scale, a_scale=None, b_scale=None):
    """fp32 reference: bf16 x times the widened codes and the per-tensor scale, plus each token's own adapter (h rounded
    to bf16 before its expand, as the kernel does)."""
    xf = x.float()
    y = xf @ (w_q.float() * s_w.float()).t()
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


_ZERO = {}   # device -> a zero int32 (N_ACTIVE of a base-only call)


def _w16_cfg(M: int, K: int) -> dict:
    """Base-only tiles by batch size, from bench/w16_bench.py --tune on the L40S (Qwen3-8B shapes; job 65831267)."""
    if M <= 16:
        return {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4}
    if M <= 64:
        return {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 4}
    if M <= 128:
        return {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 4}
    if M <= 512:
        return {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64 if K <= 4096 else 128, "num_warps": 4 if K <= 4096 else 8,
                "num_stages": 4}
    return {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3}


def w8a16_linear(x: torch.Tensor, w_q: torch.Tensor, s_w: torch.Tensor, dot_f32: bool = False) -> torch.Tensor:
    """y = x @ (s_w * widen(w_q))^T without adapters: x [M, K] bf16 (row-major), w_q [N, K] float8_e4m3fn, s_w [1] fp32.
    The base GEMM of the weight-only FP8 engine (lowbit_lora/w16_engine.py); graph-capturable (no host synchronization)."""
    M, K = x.shape
    N = w_q.shape[0]
    z = _ZERO.get(x.device)
    if z is None:
        z = _ZERO[x.device] = torch.zeros(1, dtype=torch.int32, device=x.device)
    y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
    cg = _w16_cfg(M, K)
    grid = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(N, cg["BLOCK_N"]),)
    _gemm_multi_w16[grid](
        x, w_q, s_w, s_w, x, s_w, y, z, z, z, M, N, K, 16, 1, N, N,
        x.stride(0), w_q.stride(0), 0, 0, 0, y.stride(0),
        SPLIT_K1=1, BLOCK_R=16, EVEN_K=K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"],
        BLOCK_K=cg["BLOCK_K"], GROUP_M=8, B_FP8=False, DOT_F32=dot_f32, num_warps=cg["num_warps"], num_stages=cg["num_stages"])
    return y
