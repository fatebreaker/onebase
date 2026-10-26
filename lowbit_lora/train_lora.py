"""The serving kernels in the trainer: a LoRA projection on the engine's FP8 base whose forward AND backward run our two
fused kernels (lowbit_lora/kernels/fp8_lora.py: K1 = shrink + per-token FP8 quantization, K2 = FP8 GEMM with the expand
in its epilogue), for colocated LoRA RL (RL-KERNELS, docs/PLAN_MLSYS_V2.md).

    y  = s_x s_W (Q(x) W^T) + h B^T,                 h  = s x A^T          (forward: K1 on (x, A), K2 on (Q(x), W, B))
    dX = s_g s_W (Q(g) W)   + u A,                   u  = s g B            (backward: K1 on (g, B^T), K2 on (Q(g), W^T, A^T))
    dA = u^T x,   dB = g^T h                                               (two small GEMMs, fp32 out)

g = dL/dy. The forward is the engine's arithmetic (per-token e4m3 activations, the engine's codes and scale, a bf16 LoRA path
with fp32 accumulation and h rounded to bf16 before the expand). The backward quantizes g per row to e4m3 inside K1 and
reads W's codes transposed (a per-call copy of one weight). The adapter's fp32 master weights (PEFT's lora_A / lora_B, which
the optimizer updates and save_pretrained writes) are cast to bf16 per call, as the engine holds them.

    from mlsys.matched_fp8 import patch_model; patch_model(model, parts="both", use_scaled_mm=True)
    from lowbit_lora import fp8_train; fp8_train.fp8_backward(model)       # the projections without an adapter
    model = get_peft_model(model, cfg)                                     # PEFT wraps the FP8 projections
    from lowbit_lora import train_lora; train_lora.install(model)          # adapter projections -> the fused kernels
"""
from __future__ import annotations

import torch
import triton

from lowbit_lora.kernels.fp8_lora import _gemm_lora_kernel, _shrink_absmax_kernel
from lowbit_lora.kernels.fp8_train_kernels import gemm

_CFG_S = {"BLOCK_SR": 64, "BLOCK_K": 128, "num_warps": 4, "num_stages": 3}
_CFG_G = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 8, "num_stages": 3}


def _k1(x: torch.Tensor, a: torch.Tensor, scale: float):
    """K1 in prefill mode: h = scale * x @ a^T (fp32, [M, r]), x_q (e4m3) and its per-row scales."""
    M, K = x.shape
    R = a.shape[0]
    hp = torch.empty(M * R, dtype=torch.float32, device=x.device)
    pmax = torch.empty(M, dtype=torch.float32, device=x.device)
    xq = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    sx = torch.empty(M, dtype=torch.float32, device=x.device)
    bm = 16                                    # 16-row blocks: enough programs for a trainer's M
    grid = (triton.cdiv(M, bm), triton.cdiv(R, _CFG_S["BLOCK_SR"]), 1)
    _shrink_absmax_kernel[grid](x, a, sx, hp, pmax, xq, sx, M, K, R, R, 1, float(scale), x.stride(0), a.stride(0), xq.stride(0),
                                SPLIT_K=1, WRITE_XQ=True, HAS_LORA=True, BLOCK_M=bm, BLOCK_SR=_CFG_S["BLOCK_SR"],
                                BLOCK_K=_CFG_S["BLOCK_K"], A_FP8=False, DOT_F32=False, RN=True,
                                num_warps=_CFG_S["num_warps"], num_stages=_CFG_S["num_stages"])
    return hp, xq, sx


def _k2(x: torch.Tensor, xq: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor, hp: torch.Tensor,
        b: torch.Tensor) -> torch.Tensor:
    """K2 in prefill mode: y = sx * sw * (xq @ w^T) + bf16(h) @ b^T, w [N, K] e4m3 (K contiguous), b [N, r] bf16."""
    M = xq.shape[0]
    N, K = w.shape
    R = b.shape[1]
    y = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
    cg = _CFG_G
    assert N % cg["BLOCK_N"] == 0, (N, cg["BLOCK_N"])
    grid = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(N, cg["BLOCK_N"]),)
    _gemm_lora_kernel[grid](x, xq, sx, sx, w, sw, hp, b, sw, y, M, N, K, R, 1, N, N,
                            x.stride(0), xq.stride(0), w.stride(0), b.stride(0), y.stride(0),
                            X_FP8=True, HAS_LORA=True, SPLIT_K1=1, BLOCK_R=max(16, triton.next_power_of_2(R)),
                            EVEN_K=K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"],
                            BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"], PRECISE=True, B_FP8=False, DOT_F32=False, RN=True,
                            num_warps=cg["num_warps"], num_stages=cg["num_stages"])
    return y


def k2_plain(xq: torch.Tensor, sx: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """K2 without an adapter: y = sx * sw * (xq @ w^T), w [N, K] e4m3 (K contiguous); any N (masked tiles)."""
    M = xq.shape[0]
    N, K = w.shape
    y = torch.empty(M, N, dtype=torch.bfloat16, device=xq.device)
    cg = _CFG_G
    grid = (triton.cdiv(M, cg["BLOCK_M"]) * triton.cdiv(N, cg["BLOCK_N"]),)
    _gemm_lora_kernel[grid](xq, xq, sx, sx, w, sw, sx, w, sw, y, M, N, K, 16, 1, N, N,
                            xq.stride(0), xq.stride(0), w.stride(0), 16, y.stride(0),
                            X_FP8=True, HAS_LORA=False, SPLIT_K1=1, BLOCK_R=16,
                            EVEN_K=K % cg["BLOCK_K"] == 0, BLOCK_M=cg["BLOCK_M"], BLOCK_N=cg["BLOCK_N"],
                            BLOCK_K=cg["BLOCK_K"], GROUP_M=cg["GROUP_M"], PRECISE=True, B_FP8=False, DOT_F32=False, RN=True,
                            num_warps=cg["num_warps"], num_stages=cg["num_stages"])
    return y


class _FusedLoRAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_codes, w_scale, a_master, b_master, scale: float):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        a = a_master.to(torch.bfloat16).contiguous()              # [r, K]
        b = b_master.to(torch.bfloat16).contiguous()              # [N, r]
        hp, xq, sx = _k1(x2, a, scale)
        y = gemm(xq, sx, w_codes, w_scale, hp=hp, b=b)            # K2's arithmetic, the expand in the epilogue
        h = hp.view(-1, a.shape[0]).to(torch.bfloat16)            # the bf16 h the expand used (scale included)
        ctx.save_for_backward(x2, h, w_codes, w_scale, a, b)
        ctx.scale = scale
        ctx.shape = shape
        return y.reshape(*shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, h, w_codes, w_scale, a, b = ctx.saved_tensors
        n, k = w_codes.shape
        g = gy.reshape(-1, n).to(torch.bfloat16).contiguous()
        bt = b.t().contiguous()                                   # [r, N]: K1's "A" for the backward shrink
        up, gq, sg = _k1(g, bt, ctx.scale)                        # u = scale * g @ B, and Q(g)
        at = a.t().contiguous()                                   # [K, r]: the expand matrix for dX (small)
        gx = gemm(gq, sg, w_codes, w_scale, transposed=True, hp=up, b=at)   # Q(g) W + bf16(u) A, W's codes read in place
        u = up.view(-1, a.shape[0])
        da = (u.t().to(torch.bfloat16) @ x2).float()              # [r, K] = u^T x
        db = (g.t() @ h).float()                                  # [N, r] = g^T h
        return gx.reshape(*ctx.shape[:-1], k), None, None, da, db, None


def _mm8(aq, sa, w_rows, sw):
    """CUTLASS/cuBLAS FP8 GEMM through torch._scaled_mm: aq [M, K'] e4m3 with per-row scales sa [M], w_rows [N', K'] e4m3
    (K' contiguous) with the per-tensor scale sw -> [M, N'] bf16."""
    n = w_rows.shape[0]
    return torch._scaled_mm(aq, w_rows.t(), scale_a=sa.reshape(-1, 1), scale_b=sw.reshape(1, 1).expand(1, n).contiguous(),
                            out_dtype=torch.bfloat16)


class _HybridLoRAFn(torch.autograd.Function):
    """The serving plugin's hybrid path in the trainer: K1 (shrink + per-token quantization in one pass), the vendor FP8
    GEMM (CUTLASS on Hopper) for the base, and the expand as a small bf16 GEMM added to its output; the backward alike."""

    @staticmethod
    def forward(ctx, x, w_codes, w_scale, a_master, b_master, scale: float):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        a = a_master.to(torch.bfloat16).contiguous()
        b = b_master.to(torch.bfloat16).contiguous()
        hp, xq, sx = _k1(x2, a, scale)
        h = hp.view(-1, a.shape[0]).to(torch.bfloat16)
        y = torch.addmm(_mm8(xq, sx, w_codes, w_scale), h, b.t())
        ctx.save_for_backward(x2, h, w_codes, w_scale, a, b)
        ctx.scale, ctx.shape = scale, shape
        return y.reshape(*shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, gy):
        x2, h, w_codes, w_scale, a, b = ctx.saved_tensors
        n, k = w_codes.shape
        g = gy.reshape(-1, n).to(torch.bfloat16).contiguous()
        up, gq, sg = _k1(g, b.t().contiguous(), ctx.scale)
        u = up.view(-1, a.shape[0])
        gx = torch.addmm(_mm8(gq, sg, w_codes.t().contiguous(), w_scale), u.to(torch.bfloat16), a)
        da = (u.t().to(torch.bfloat16) @ x2).float()
        db = (g.t() @ h).float()
        return gx.reshape(*ctx.shape[:-1], k), None, None, da, db, None


class _W16LoRAFn(torch.autograd.Function):
    """A LoRA projection with the weight-matched W8A16 arithmetic: y = x (s_W W)^T + h B^T, h = s x A^T in bf16 (as the
    engine's LoRA path), through the mixed-input GEMM with the expand in its epilogue; dX = g (s_W W) + u A, u = s g B."""

    @staticmethod
    def forward(ctx, x, w_codes, w_scale, a_master, b_master, scale: float):
        from lowbit_lora.kernels.fp8_train_kernels import gemm_w8a16
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous()
        a = a_master.to(torch.bfloat16).contiguous()
        b = b_master.to(torch.bfloat16).contiguous()
        h = ((x2 @ a.t()).float() * scale).to(torch.bfloat16)
        y = gemm_w8a16(x2, w_codes, w_scale, hp=h, b=b)
        ctx.save_for_backward(x2, h, w_codes, w_scale, a, b)
        ctx.scale, ctx.shape = scale, shape
        return y.reshape(*shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, gy):
        from lowbit_lora.kernels.fp8_train_kernels import gemm_w8a16
        x2, h, w_codes, w_scale, a, b = ctx.saved_tensors
        n, k = w_codes.shape
        g = gy.reshape(-1, n).to(torch.bfloat16).contiguous()
        u = ((g @ b).float() * ctx.scale).to(torch.bfloat16)
        gx = gemm_w8a16(g, w_codes, w_scale, transposed=True, hp=u, b=a.t().contiguous())
        da = (u.t() @ x2).float()
        db = (g.t() @ h).float()
        return gx.reshape(*ctx.shape[:-1], k), None, None, da, db, None


def install(model: torch.nn.Module, kind: str = "fused") -> dict:
    """Route every PEFT LoRA projection whose base layer holds the engine's FP8 codes through our kernels (one active
    adapter "default", no dropout): kind "fused" (K1 + K2, the GEMM with the expand in its epilogue) or "hybrid" (K1 +
    the vendor FP8 GEMM + a bf16 expand). The PEFT parameters stay the trainable ones, so saving is unchanged."""
    fn = {"fused": _FusedLoRAFn, "hybrid": _HybridLoRAFn, "w16": _W16LoRAFn}[kind]
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for m in model.modules():
        if isinstance(m, LoraLinear) and hasattr(m.base_layer, "w_codes"):
            base = m.base_layer
            la, lb = m.lora_A["default"], m.lora_B["default"]
            sc = float(m.scaling["default"])
            bias = base.bias          # frozen (Qwen2's q/k/v projections have one): added after the fused op, no gradient

            def forward(self, x, *args, _base=base, _la=la, _lb=lb, _sc=sc, _fn=fn, _bias=bias, **kwargs):
                y = _fn.apply(x, _base.w_codes, _base.w_scale, _la.weight, _lb.weight, _sc)
                return y if _bias is None else y + _bias.to(y.dtype)
            m.forward = forward.__get__(m, type(m))
            n += 1
    return {f"{kind}_lora_layers": n}
