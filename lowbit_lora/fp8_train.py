"""FP8 trainer GEMMs for colocated LoRA RL: the weight-matched trainer's base GEMMs on the engine's FP8 codes in the
forward AND the backward.

mlsys.matched_fp8.patch_model computes the engine's codes and scales (vLLM's online FP8: e4m3 weights, one per-tensor
scale per fused group) and offers a W8A8 forward through torch._scaled_mm with per-token activation scales (parts="both").
Its backward multiplies the incoming gradient by the dequantized bf16 weight, materialized on every call, and the RL runs
so far used parts="w" (FP8 weights dequantized to bf16 in the forward too). Here the backward is an FP8 GEMM as well:

    dX = Q(dY) @ W_codes * s_dY * s_W

with dY quantized per row (per token) to FP8 (e4m3 by default; e5m2 optional) and the codes transposed per call: FP8 GEMMs
on Hopper and Ada take both operands with the reduction dimension contiguous, and the reduction of dX runs over W's output
dimension, so the backward needs W^T's codes. The transposed copy lives for one call (one weight at a time), so the trainer
keeps holding one byte per base parameter. The forward is matched_fp8's W8A8 unchanged (the engine's own math), so the
trainer--engine agreement is that of parts="both"; only the gradient's arithmetic changes. The weight is frozen (LoRA),
so no weight gradient is computed.

    from mlsys.matched_fp8 import patch_model
    from lowbit_lora.fp8_train import fp8_backward
    patch_model(model, parts="both", use_scaled_mm=True); fp8_backward(model, fmt="e4m3")   # before PEFT wraps the model
"""
from __future__ import annotations

import os

import torch
import triton

from mlsys import matched_fp8 as mf

E5M2_MAX = 57344.0
_STATE = {"bwd_calls": 0, "fallbacks": 0, "error": None}
# per-row FP8 quantization: "triton" = our K1 kernel without the LoRA shrink (one bf16 read and one e4m3 write per row,
# the engine's rounding and scale floor), "torch" = matched_fp8's PyTorch version (TRAINER-BENCH v1, fp32 temporaries)
QUANT = os.environ.get("LOWBIT_TRAIN_QUANT", "triton")
# the GEMM: "torch" = torch._scaled_mm (CUTLASS on Hopper), "triton" = our K2 without an adapter (PyTorch's row-wise FP8 GEMM is
# slow on Ada: TRAINER-BENCH v2, L40S); "auto" picks triton below compute capability 9.0 and torch at 9.0
GEMM = os.environ.get("LOWBIT_TRAIN_GEMM", "auto")


def _use_triton(dev) -> bool:
    return GEMM == "triton" or (GEMM == "auto" and torch.cuda.get_device_capability(dev) != (9, 0))


def quant_rows_triton(x2: torch.Tensor):
    """e4m3 codes and per-row scales of a contiguous bf16 [M, K] tensor through K1 (HAS_LORA=False, WRITE_XQ=True)."""
    from lowbit_lora.kernels.fp8_lora import _shrink_absmax_kernel
    M, K = x2.shape
    xq = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x2.device)
    sx = torch.empty(M, dtype=torch.float32, device=x2.device)
    pmax = torch.empty(M, dtype=torch.float32, device=x2.device)
    bm = 16                                    # 16-row blocks: enough programs for a trainer's M (GEMM-BENCH: 64 left SMs idle)
    _shrink_absmax_kernel[(triton.cdiv(M, bm), 1, 1)](
        x2, x2, sx, sx, pmax, xq, sx, M, K, 1, 1, 1, 1.0, x2.stride(0), 0, xq.stride(0),
        SPLIT_K=1, WRITE_XQ=True, HAS_LORA=False, BLOCK_M=bm, BLOCK_SR=16, BLOCK_K=128, A_FP8=False, DOT_F32=False, RN=True,
        num_warps=4, num_stages=3)
    return xq, sx[:, None]


def quant_rows(x2: torch.Tensor):
    if QUANT == "triton" and x2.is_cuda:
        return quant_rows_triton(x2.to(torch.bfloat16).contiguous())
    return mf._quant_rows(x2)


def _quant_rows(x: torch.Tensor, fmt: str):
    """Per-row dynamic FP8 quantization (as matched_fp8._quant_rows for e4m3; e5m2 with its own range)."""
    if fmt == "e4m3":
        return quant_rows(x)
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    s = torch.clamp(amax / E5M2_MAX, min=1e-30)
    return torch.clamp(x.float() / s, -E5M2_MAX, E5M2_MAX).to(torch.float8_e5m2), s


class _FP8LinearFB(torch.autograd.Function):
    """W8A8 forward (matched_fp8's) and FP8 backward for a frozen base weight."""

    @staticmethod
    def forward(ctx, x, w_codes, w_scale, fmt: str):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = quant_rows(x2)
        n = w_codes.shape[0]
        if _use_triton(x.device):
            from lowbit_lora.kernels.fp8_train_kernels import gemm
            y = gemm(xq, sx.reshape(-1).contiguous(), w_codes, w_scale)
        else:
            y = torch._scaled_mm(xq, w_codes.t(), scale_a=sx, scale_b=w_scale.reshape(1, 1).expand(1, n).contiguous(),
                                 out_dtype=torch.bfloat16)
        ctx.save_for_backward(w_codes, w_scale)
        ctx.fmt = fmt
        return y.reshape(*shape[:-1], n)

    @staticmethod
    def backward(ctx, gy):
        w_codes, w_scale = ctx.saved_tensors
        n, k = w_codes.shape
        g2 = gy.reshape(-1, n)
        _STATE["bwd_calls"] += 1
        try:
            gq, sg = _quant_rows(g2, ctx.fmt)
            if _use_triton(gy.device) and ctx.fmt == "e4m3":   # our GEMM reads the codes in place: no transposed copy
                from lowbit_lora.kernels.fp8_train_kernels import gemm
                gx = gemm(gq, sg.reshape(-1).contiguous(), w_codes, w_scale, transposed=True)
            else:
                wt = w_codes.t().contiguous()                      # [k, n]: n (the reduction) contiguous
                gx = torch._scaled_mm(gq, wt.t(), scale_a=sg, scale_b=w_scale.reshape(1, 1).expand(1, k).contiguous(),
                                      out_dtype=torch.bfloat16)
        except RuntimeError as e:                                  # unsupported here: matched_fp8's bf16 backward, loudly once
            if _STATE["error"] is None:
                _STATE["error"] = str(e)[:300]
                print(f"[fp8_train] FP8 backward unusable, bf16 fallback: {_STATE['error']}", flush=True)
            _STATE["fallbacks"] += 1
            w = (w_codes.to(torch.float32) * w_scale).to(torch.bfloat16)
            gx = g2.to(torch.bfloat16) @ w
        return gx.reshape(*gy.shape[:-1], k), None, None, None


class _W8A16LinearFn(torch.autograd.Function):
    """The weight-matched trainer's arithmetic (matched_fp8 parts="w": the engine's FP8 weights, bf16 activations) through our
    mixed-input GEMM: the codes widened to bf16 in registers, never materialized; the backward reads them in place."""

    @staticmethod
    def forward(ctx, x, w_codes, w_scale):
        from lowbit_lora.kernels.fp8_train_kernels import gemm_w8a16
        shape = x.shape
        y = gemm_w8a16(x.reshape(-1, shape[-1]).to(torch.bfloat16).contiguous(), w_codes, w_scale)
        ctx.save_for_backward(w_codes, w_scale)
        return y.reshape(*shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, gy):
        from lowbit_lora.kernels.fp8_train_kernels import gemm_w8a16
        w_codes, w_scale = ctx.saved_tensors
        n, k = w_codes.shape
        gx = gemm_w8a16(gy.reshape(-1, n).to(torch.bfloat16).contiguous(), w_codes, w_scale, transposed=True)
        return gx.reshape(*gy.shape[:-1], k), None, None


def fp8_backward(model: torch.nn.Module, fmt: str = "e4m3", kind: str = "w8a8") -> dict:
    """Give every module that mlsys.matched_fp8.patch_model rewrote our forward and backward: kind "w8a8" (W8A8 forward
    as parts="both", FP8 backward) or "w8a16" (parts="w"'s arithmetic through the mixed-input GEMM)."""
    assert fmt in ("e4m3", "e5m2") and kind in ("w8a8", "w8a16")
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and hasattr(m, "w_codes"):
            if kind == "w8a8":
                def forward(self, x, _fmt=fmt):
                    y = _FP8LinearFB.apply(x, self.w_codes, self.w_scale, _fmt)
                    return y + self.bias if self.bias is not None else y
            else:
                def forward(self, x):
                    y = _W8A16LinearFn.apply(x, self.w_codes, self.w_scale)
                    return y + self.bias if self.bias is not None else y
            m.forward = forward.__get__(m, type(m))
            n += 1
    return {"fp8_backward_layers": n, "grad_format": fmt, "kind": kind}


def stats() -> dict:
    return {**_STATE, "quant": QUANT, "gemm": GEMM}
