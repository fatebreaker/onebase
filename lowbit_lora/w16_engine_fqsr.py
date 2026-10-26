"""The fake-quant control with stochastic rounding (ICML plan v2, E1 in docs/PLAN_ICML_V2.md): w16_engine_fq's patch, except
that each FP8 layer's bf16 activations are rounded per token to e4m3 *stochastically* (to the e4m3 neighbor above with
probability (y - lo) / (hi - lo), fresh noise at every call) instead of to the nearest code, then multiplied by the e4m3 weight
codes through the same W8A16 GEMM. With round-to-nearest the engine's distribution is a fixed, biased copy of the trainer's
(MISMATCH-DECOMP-2: activation rounding is 85-106% of the W8A8 engine's excess KL); with stochastic rounding the rounding
error has mean zero, so the engine's distribution averaged over the noise matches the trainer's to second order.
Run the engine eager (enforce_eager=True): the noise comes from torch.rand at every call.

    from lowbit_lora import w16_engine_fqsr; w16_engine_fqsr.install()   # before vllm.LLM(...), in-process, LOWBIT_FUSED=0
A separate module: w16_engine.py and w16_engine_fq.py, which registered runs use, are unchanged.
"""
from __future__ import annotations

import os

import torch

_INSTALLED = False
stats = {"w16fqsr_layers": 0, "fallback_layers": 0}
_FP8_MAX = 448.0
_MIN_SCALE = 1.0 / (448.0 * 512.0)


def _code_value(mag: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return ((mag | (neg.to(torch.int16) << 7)).to(torch.uint8)).view(torch.float8_e4m3fn).float()


def e4m3_neighbors(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The e4m3 values just below and just above y (fp32, |y| <= 448): round to nearest, then step one code toward y."""
    q = y.to(torch.float8_e4m3fn)
    qf = q.float()
    code = q.view(torch.uint8).to(torch.int16)
    neg = (code & 0x80) != 0
    mag = code & 0x7F
    up_mag = torch.clamp(mag + 1, max=0x7E)              # 0x7E = 448, the largest finite code; 0x7F is NaN
    dn_mag = torch.clamp(mag - 1, min=0)
    up = torch.where(neg, _code_value(dn_mag, neg), _code_value(up_mag, neg))     # toward +inf
    dn = torch.where(neg, _code_value(up_mag, neg), _code_value(dn_mag, neg))     # toward -inf
    zneg = neg & (mag == 0)                              # -0 steps up to the smallest positive code
    up = torch.where(zneg, _code_value(torch.ones_like(mag), torch.zeros_like(neg)), up)
    lo = torch.where(qf <= y, qf, dn)
    hi = torch.where(qf <= y, torch.where(qf == y, qf, up), qf)
    return lo, hi


def sr_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-token (row) stochastic FP8 e4m3 round trip of a bf16 [M, K] tensor (vLLM's per-token scale rule)."""
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp(min=_MIN_SCALE)
    y = (xf / s).clamp(-_FP8_MAX, _FP8_MAX)
    lo, hi = e4m3_neighbors(y)
    gap = hi - lo
    p = torch.where(gap > 0, (y - lo) / torch.where(gap > 0, gap, torch.ones_like(gap)), torch.zeros_like(gap))
    r = torch.where(torch.rand_like(p) < p, hi, lo)
    return (r * s).to(torch.bfloat16)


def _linear(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    from lowbit_lora.kernels.w8a16_lora_multi import w8a16_linear
    return w8a16_linear(sr_rows(x), w, sw)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        raise RuntimeError("w16_engine_fqsr patches this process: set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in it")
    if os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"):
        raise RuntimeError("w16_engine_fqsr: LOWBIT_FUSED=0 (vLLM's LoRA kernels; the control changes only the base GEMM's input)")
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="lowbit_w8a16fqsr_linear", op_func=_linear, mutates_args=[], fake_impl=_linear_fake)
    orig_process = Fp8PerTensorOnlineLinearMethod.process_weights_after_loading
    orig_apply = Fp8PerTensorOnlineLinearMethod.apply

    def process(self, layer):
        orig_process(self, layer)
        ok = (isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel) and layer.weight.dtype == torch.float8_e4m3fn
              and layer.weight_scale.numel() == 1 and layer.weight.t().is_contiguous())
        layer._lowbit_w16fqsr = ok
        stats["w16fqsr_layers" if ok else "fallback_layers"] += 1

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_lowbit_w16fqsr", False) or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.vllm.lowbit_w8a16fqsr_linear(x2, layer.weight.t(), layer.weight_scale.reshape(1))
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    Fp8PerTensorOnlineLinearMethod.process_weights_after_loading = process
    Fp8PerTensorOnlineLinearMethod.apply = apply
    _INSTALLED = True
    print("[lowbit] w16 stochastic-rounding fake-quant engine installed: per-token FP8 activations rounded stochastically, "
          "W8A16 GEMM", flush=True)
