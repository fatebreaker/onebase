"""The fake-quant control for the matched-precision engine (MISMATCH-DECOMP-2; the v3 mock review, R2 M3): w16_engine's patch,
except that each FP8 layer's bf16 activations are first rounded per token to e4m3 and back (scale max(amax / 448,
1 / (448 * 512)), vLLM's dynamic per-token rule, as K1 computes it), then multiplied by the e4m3 weight codes through the
same W8A16 GEMM. Its KL to the twin W8A16 trainer isolates the activations' rounding from the W8A8 path's other changes
(FP8 tensor-core MMA, its accumulation, CUTLASS's epilogue).

    from lowbit_lora import w16_engine_fq; w16_engine_fq.install()   # before vllm.LLM(...), in-process, LOWBIT_FUSED=0
A separate module so that lowbit_lora/w16_engine.py, which registered runs use (RL-MATCHED), is unchanged.
"""
from __future__ import annotations

import os

import torch

_INSTALLED = False
stats = {"w16fq_layers": 0, "fallback_layers": 0}
_FP8_MAX = 448.0
_MIN_SCALE = 1.0 / (448.0 * 512.0)


def fake_quant_rows(x: torch.Tensor) -> torch.Tensor:
    """Per-token (row) FP8 e4m3 round trip of a bf16 [M, K] tensor."""
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / _FP8_MAX).clamp(min=_MIN_SCALE)
    return ((xf / s).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).float() * s).to(torch.bfloat16)


def _linear(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    from lowbit_lora.kernels.w8a16_lora_multi import w8a16_linear
    return w8a16_linear(fake_quant_rows(x), w, sw)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        raise RuntimeError("w16_engine_fq patches this process: set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in it")
    if os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"):
        raise RuntimeError("w16_engine_fq: LOWBIT_FUSED=0 (vLLM's LoRA kernels; the control changes only the base GEMM's input)")
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="lowbit_w8a16fq_linear", op_func=_linear, mutates_args=[], fake_impl=_linear_fake)
    orig_process = Fp8PerTensorOnlineLinearMethod.process_weights_after_loading
    orig_apply = Fp8PerTensorOnlineLinearMethod.apply

    def process(self, layer):
        orig_process(self, layer)
        ok = (isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel) and layer.weight.dtype == torch.float8_e4m3fn
              and layer.weight_scale.numel() == 1 and layer.weight.t().is_contiguous())
        layer._lowbit_w16fq = ok
        stats["w16fq_layers" if ok else "fallback_layers"] += 1

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_lowbit_w16fq", False) or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.vllm.lowbit_w8a16fq_linear(x2, layer.weight.t(), layer.weight_scale.reshape(1))
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    Fp8PerTensorOnlineLinearMethod.process_weights_after_loading = process
    Fp8PerTensorOnlineLinearMethod.apply = apply
    _INSTALLED = True
    print("[lowbit] w16 fake-quant engine installed: per-token FP8 activations, W8A16 GEMM", flush=True)
