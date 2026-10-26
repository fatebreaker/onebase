"""Fused FP8 GEMM + LoRA, the Phase-0 prototype (docs/REGISTRY.md, P0-KERNEL). One adapter; Triton.

    y = s_x * s_w * (x_q @ W_q^T)  +  h_s @ B_s^T,      h = scale * x @ A^T

x [M, K] bf16 is quantized per token to FP8 E4M3 (s_x = max|x_m| / 448), W_q [N, K] is FP8 with one float32 scale
s_w (vLLM's online FP8), A [S*r, K] stacks the LoRA A of the S slices of a fused projection (q/k/v or gate/up), and
B [N, r] holds each output channel's LoRA B row; output column n belongs to slice s(n). The shrink uses the bf16 x
with fp32 accumulation and h is rounded to bf16 before the expand, as vLLM's Punica kernels do.

Two kernels instead of vLLM's four (quant, GEMM, shrink, expand):
  _shrink_absmax  split-K partial sums of h ([SPLIT_K, S, M, r] fp32, layout chosen so that SPLIT_K = 1 is exactly
                  vLLM's shrink buffer [S, M, r]) and partial per-token maxima |x| ([SPLIT_K, M]). With XQ (prefill),
                  SPLIT_K = 1 and a second pass over K writes x_q and s_x.
  _gemm_lora      the FP8 GEMM. Decode: x is quantized on the fly from the partial maxima (x is small, W dominates).
                  Prefill: reads x_q. The epilogue sums the h partials of its rows and slice and adds h_s @ B_s^T
                  before the single bf16 store, so y is written once and never re-read.
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


def _shrink_configs():
    out = []
    for bm in (16, 64):
        for bsr in (64, 128):
            for bk in (64, 128, 256):
                for nw in (4, 8):
                    out.append(triton.Config({"BLOCK_M": bm, "BLOCK_SR": bsr, "BLOCK_K": bk}, num_warps=nw,
                                             num_stages=3))
    return out


@triton.jit
def _shrink_absmax_kernel(X, A, SA, HP, PMAX, XQ, SX, M, K, SR, R, S, scale,
                          stride_xm, stride_am, stride_xqm,
                          SPLIT_K: tl.constexpr, WRITE_XQ: tl.constexpr, HAS_LORA: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_SR: tl.constexpr, BLOCK_K: tl.constexpr,
                          A_FP8: tl.constexpr = False, DOT_F32: tl.constexpr = False, RN: tl.constexpr = True):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_r * BLOCK_SR + tl.arange(0, BLOCK_SR)
    mm = offs_m < M
    mc = offs_c < SR
    k_per = tl.cdiv(tl.cdiv(K, SPLIT_K), BLOCK_K) * BLOCK_K
    k_lo = pid_k * k_per
    k_hi = tl.minimum(k_lo + k_per, K)
    acc = tl.zeros((BLOCK_M, BLOCK_SR), dtype=tl.float32)
    amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mk = offs_k < k_hi
        x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if HAS_LORA:
            a = tl.load(A + offs_c[:, None] * stride_am + offs_k[None, :], mask=mc[:, None] & mk[None, :],
                        other=0.0)
            if A_FP8:     # FP8-stored A: exact upcast; its per-row scale is applied once, after the reduction
                a = a.to(tl.bfloat16)
            if DOT_F32:   # CPU interpreter only: its bf16/fp8 tl.dot is wrong
                acc = tl.dot(x.to(tl.float32), tl.trans(a.to(tl.float32)), acc)
            else:
                acc = tl.dot(x, tl.trans(a), acc)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32)), axis=1))
    if HAS_LORA:   # h partial, layout [SPLIT_K, S, M, R]
        s_idx = offs_c // R
        r_idx = offs_c % R
        hp = HP + pid_k * (S * M * R) + s_idx[None, :] * (M * R) + offs_m[:, None] * R + r_idx[None, :]
        if A_FP8:
            acc = acc * tl.load(SA + offs_c, mask=mc, other=0.0)[None, :]
        tl.store(hp, acc * scale, mask=mm[:, None] & mc[None, :])
    if pid_r == 0:
        tl.store(PMAX + pid_k * M + offs_m, amax, mask=mm)
    if WRITE_XQ:   # SPLIT_K == 1: every program saw its rows' full maxima; the column programs share the x_q pass
        sx = tl.maximum(tl.math.div_rn(amax, _FP8_MAX), _MIN_SCALE)
        if pid_r == 0:
            tl.store(SX + offs_m, sx, mask=mm)
        n_r = tl.num_programs(1)
        k_slice = tl.cdiv(tl.cdiv(K, n_r), BLOCK_K) * BLOCK_K
        q_lo = pid_r * k_slice
        q_hi = tl.minimum(q_lo + k_slice, K)
        for k0 in range(q_lo, q_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            msk = mm[:, None] & (offs_k[None, :] < q_hi)
            x = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :], mask=msk, other=0.0).to(tl.float32)
            q = to_e4m3(tl.clamp(tl.math.div_rn(x, sx[:, None]), -_FP8_MAX, _FP8_MAX), RN)
            tl.store(XQ + offs_m[:, None] * stride_xqm + offs_k[None, :], q, mask=msk)


def _gemm_configs():
    out = []
    for bm, bn, bk, nw, ns in [(16, 32, 256, 4, 4), (16, 64, 128, 4, 4), (16, 64, 256, 4, 3), (16, 128, 128, 4, 4),
                               (32, 64, 128, 4, 4), (32, 128, 128, 4, 4), (64, 64, 128, 4, 4), (64, 128, 128, 4, 4),
                               (64, 128, 128, 8, 4), (64, 256, 128, 8, 3), (128, 128, 128, 8, 4),
                               (128, 256, 64, 8, 4), (128, 256, 128, 8, 3), (128, 128, 64, 4, 4)]:
        out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=nw,
                                 num_stages=ns))
    return out


@triton.jit
def _gemm_lora_kernel(X, XQ, SX, PMAX, W, SW, HP, B, SB, Y, M, N, K, R, S, END0, END1,
                      stride_xm, stride_xqm, stride_wn, stride_bn, stride_ym,
                      X_FP8: tl.constexpr, HAS_LORA: tl.constexpr, SPLIT_K1: tl.constexpr, BLOCK_R: tl.constexpr,
                      EVEN_K: tl.constexpr,
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
    sw = tl.load(SW)
    acc = acc * (sx[:, None] * sw)
    if HAS_LORA:
        n0 = pid_n * BLOCK_N
        s = (n0 >= END0).to(tl.int32) + (n0 >= END1).to(tl.int32)   # a tile never straddles two slices
        offs_r = tl.arange(0, BLOCK_R)
        mr = offs_r < R
        h = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)
        for j in tl.static_range(SPLIT_K1):
            h += tl.load(HP + j * (S * M * R) + s * (M * R) + offs_m[:, None] * R + offs_r[None, :],
                         mask=mm[:, None] & mr[None, :], other=0.0)
        bt = tl.load(B + offs_n[None, :] * stride_bn + offs_r[:, None], mask=mn[None, :] & mr[:, None], other=0.0)
        if B_FP8:         # FP8-stored B: exact upcast; the per-output-channel scale multiplies the expand's columns
            bt = bt.to(tl.bfloat16)
            if DOT_F32:
                lo = tl.dot(h.to(tl.bfloat16).to(tl.float32), bt.to(tl.float32))
            else:
                lo = tl.dot(h.to(tl.bfloat16), bt)
            acc += lo * tl.load(SB + offs_n, mask=mn, other=0.0)[None, :]
        elif DOT_F32:
            acc = tl.dot(h.to(tl.bfloat16).to(tl.float32), bt.to(tl.float32), acc)
        else:
            acc = tl.dot(h.to(tl.bfloat16), bt, acc)
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])


class FusedFP8LoRA:
    """One fused projection with static buffers, so that a call can be captured in a CUDA graph.

    w_q [N, K] float8_e4m3fn, s_w [1] float32, a [S*r, K] bf16, b [N, r] bf16, slices: output sizes per slice.
    mode: "decode" (on-the-fly x quantization, split-K shrink) or "prefill" (x_q written by the shrink kernel).
    Kernel configs are fixed per instance (set by tune(), or the defaults), so graph capture never autotunes.
    """

    def __init__(self, w_q, s_w, a, b, slices, scale, max_m, *, split_k=None, has_lora=True, a_scale=None,
                 b_scale=None):
        self.w_q, self.s_w, self.a, self.b = w_q, s_w, a, b
        self.a_fp8, self.b_fp8 = a_scale is not None, b_scale is not None     # FP8-stored adapter halves of the path
        self.a_scale = a_scale if a_scale is not None else s_w                  # (dummy pointer when unused)
        self.b_scale = b_scale if b_scale is not None else s_w
        self.N, self.K = w_q.shape
        self.S = len(slices)
        self.R = b.shape[1]
        assert a.shape == (self.S * self.R, self.K) and sum(slices) == self.N and self.S <= 3
        ends = [sum(slices[:i + 1]) for i in range(self.S)]
        self.end0 = ends[0] if self.S >= 2 else self.N
        self.end1 = ends[1] if self.S >= 3 else self.N
        self.slices = tuple(slices)
        self.scale = float(scale)
        self.has_lora = has_lora
        self.split_k = split_k or max(1, min(16, self.K // 512))
        dev = w_q.device
        self.hp = torch.empty(self.split_k * self.S * max_m * self.R, dtype=torch.float32, device=dev)
        self.pmax = torch.empty(self.split_k * max_m, dtype=torch.float32, device=dev)
        self.xq = torch.empty(max_m, self.K, dtype=torch.float8_e4m3fn, device=dev)
        self.sx = torch.empty(max_m, dtype=torch.float32, device=dev)
        self.y = torch.empty(max_m, self.N, dtype=torch.bfloat16, device=dev)
        self.cfg_shrink: dict = {}
        self.cfg_gemm: dict = {}
        self.dot_f32 = False          # True only under TRITON_INTERPRET=1 (tests on CPU)

    def _block_r(self) -> int:
        return max(16, triton.next_power_of_2(self.R))

    def _cfgs(self, M: int, mode: str, cfg_s: dict | None, cfg_g: dict | None) -> tuple[dict, dict]:
        cs = cfg_s or self.cfg_shrink.get((M, mode)) or {"BLOCK_M": 16 if M <= 16 else 64, "BLOCK_SR": 64,
                                                         "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}
        cg = cfg_g or self.cfg_gemm.get((M, mode)) or ({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8,
                                                        "num_warps": 4, "num_stages": 4} if M <= 64 else
                                                       {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8,
                                                        "num_warps": 8, "num_stages": 3})
        return cs, cg

    def shrink(self, x: torch.Tensor, mode: str, cfg_s: dict | None = None) -> None:
        """Kernel 1: h partials and |x| maxima; in prefill mode (SPLIT_K = 1) also x_q and s_x."""
        M = x.shape[0]
        prefill = mode == "prefill"
        split = 1 if prefill else self.split_k
        cs, _ = self._cfgs(M, mode, cfg_s, None)
        SR = self.S * self.R
        grid_s = (triton.cdiv(M, cs["BLOCK_M"]), triton.cdiv(SR, cs["BLOCK_SR"]) if self.has_lora else 1, split)
        _shrink_absmax_kernel[grid_s](
            x, self.a, self.a_scale, self.hp, self.pmax, self.xq, self.sx, M, self.K, SR, self.R, self.S, self.scale,
            x.stride(0), self.a.stride(0), self.xq.stride(0), SPLIT_K=split, WRITE_XQ=prefill,
            HAS_LORA=self.has_lora, BLOCK_M=cs["BLOCK_M"], BLOCK_SR=cs["BLOCK_SR"], BLOCK_K=cs["BLOCK_K"],
            A_FP8=self.a_fp8, DOT_F32=self.dot_f32, RN=not self.dot_f32, num_warps=cs["num_warps"], num_stages=cs["num_stages"])

    def gemm(self, x: torch.Tensor, mode: str, cfg_g: dict | None = None) -> torch.Tensor:
        """Kernel 2: the FP8 GEMM with the expand in its epilogue (reads what shrink() wrote)."""
        M = x.shape[0]
        prefill = mode == "prefill"
        split = 1 if prefill else self.split_k
        _, cg = self._cfgs(M, mode, None, cfg_g)
        assert all(sz % cg["BLOCK_N"] == 0 for sz in self.slices), (self.slices, cg["BLOCK_N"])
        y = self.y[:M]
        grid_g = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(self.N, cg["BLOCK_N"]),)
        _gemm_lora_kernel[grid_g](
            x, self.xq, self.sx, self.pmax, self.w_q, self.s_w, self.hp, self.b, self.b_scale, y, M, self.N, self.K,
            self.R, self.S,
            self.end0, self.end1, x.stride(0), self.xq.stride(0), self.w_q.stride(0), self.b.stride(0), y.stride(0),
            X_FP8=prefill, HAS_LORA=self.has_lora, SPLIT_K1=split, BLOCK_R=self._block_r(),
            EVEN_K=self.K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"], BLOCK_K=cg["BLOCK_K"],
            GROUP_M=cg["GROUP_M"], PRECISE=os.environ.get("LOWBIT_PRECISE", "1") != "0", B_FP8=self.b_fp8, DOT_F32=self.dot_f32, RN=not self.dot_f32, num_warps=cg["num_warps"],
            num_stages=cg["num_stages"])
        return y

    def __call__(self, x: torch.Tensor, mode: str, cfg_s: dict | None = None, cfg_g: dict | None = None):
        self.shrink(x, mode, cfg_s)
        return self.gemm(x, mode, cfg_g)

    def tune(self, x: torch.Tensor, mode: str, rep: int = 30) -> tuple[dict, dict]:
        """Pick the fastest shrink config, then the fastest GEMM config, for this M and mode (eager timing)."""
        from triton.testing import do_bench
        M = x.shape[0]
        best_s, best_g = None, None
        t_best = float("inf")
        for c in _shrink_configs():
            cs = {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}
            if cs["BLOCK_M"] > max(16, triton.next_power_of_2(M)) * 4:
                continue
            try:
                t = do_bench(lambda: self(x, mode, cfg_s=cs), rep=rep)
            except Exception:   # noqa: BLE001 -- out of shared memory etc.: skip the config
                continue
            if t < t_best:
                t_best, best_s = t, cs
        t_best = float("inf")
        for c in _gemm_configs():
            cg = {**c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}
            if any(sz % cg["BLOCK_N"] for sz in self.slices):
                continue
            if cg["BLOCK_M"] > max(16, triton.next_power_of_2(M)) * 2:
                continue
            try:
                t = do_bench(lambda: self(x, mode, cfg_s=best_s, cfg_g=cg), rep=rep)
            except Exception:   # noqa: BLE001
                continue
            if t < t_best:
                t_best, best_g = t, cg
        self.cfg_shrink[(M, mode)] = best_s
        self.cfg_gemm[(M, mode)] = best_g
        return best_s, best_g

    def shrink_buffer(self, M: int) -> torch.Tensor:
        """h after a prefill-mode call (SPLIT_K = 1): exactly vLLM's shrink buffer [S, M, r] (fp32)."""
        return self.hp[: self.S * M * self.R].view(self.S, M, self.R)


def quantize_rows_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 with one float32 scale per row (max |row| -> 448): (t_q, scale [rows])."""
    s = (t.float().abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    return (t.float() / s[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def reference(x, w_q, s_w, a, b, slices, scale, a_scale=None, b_scale=None):
    """fp32 reference: per-token FP8 quantization of x as the kernels do it, dequantized GEMM, exact LoRA (h in bf16)."""
    if a_scale is not None:
        a = a.float() * a_scale[:, None]
    if b_scale is not None:
        b = b.float() * b_scale[:, None]
    xf = x.float()
    sx = (xf.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1.0 / (FP8_MAX * 512.0))
    xq = (xf / sx).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * sx
    y = xq @ (w_q.float() * s_w.float()).t()
    R = b.shape[1]
    h = (xf @ a.float().t() * scale).to(torch.bfloat16).float()
    off = 0
    for s, n in enumerate(slices):
        y[:, off:off + n] += h[:, s * R:(s + 1) * R] @ b[off:off + n].float().t()
        off += n
    return y
