"""The matched-precision rollout engine (RL-MATCHED-KERNELS, docs/PLAN_MLSYS_V2.md): an in-process patch that makes vLLM's
online FP8 layers compute weight-only, W8A16, through our kernel (kernels/w8a16_lora_multi.w8a16_linear): bf16 activations
times the engine's own e4m3 codes widened in registers, fp32 accumulation, the per-tensor scale once at the end -- what the
weight-matched trainer's W8A16 kernel computes from the same codes. vLLM's online quantization, weight storage and memory
are unchanged (the codes CUTLASS would read); only the GEMM changes, so no activation is rounded. LoRA layers keep vLLM's
own adapter kernels (their base GEMM is this one), or, with the plugin in its hybrid mode (LOWBIT_FUSED=1,
LOWBIT_DISPATCH=hybrid), ours: the plugin's base_fn is quant_method.apply, i.e. this GEMM, and its kernels add the adapters.

    from lowbit_lora import w16_engine; w16_engine.install()      # before vllm.LLM(...), with the engine in this process
                                                                  # (VLLM_ENABLE_V1_MULTIPROCESSING=0) and LOWBIT_FUSED=0
Nothing here runs unless install() is called; the vLLM plugin (lowbit_lora/plugin.py) is a separate path and untouched.
"""
from __future__ import annotations

import os

import torch

_INSTALLED = False
stats = {"w16_layers": 0, "fallback_layers": 0}


def _linear(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    from lowbit_lora.kernels.w8a16_lora_multi import w8a16_linear
    return w8a16_linear(x, w, sw)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)


def install() -> None:
    if _INSTALLED:
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        raise RuntimeError("w16_engine patches this process: set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in it")
    _install()


def install_plugin() -> None:
    """vLLM general-plugin hook (pylib_w16/lowbit_w16-*.dist-info's entry point; CO-SERVE-MATCHED): with LOWBIT_W16_ENGINE=1 every
    vLLM process patches itself, so a `vllm serve` engine core, which runs in its own process, computes W8A16 too (install()'s
    in-process check is for callers that build the engine in their own process). A no-op without LOWBIT_W16_ENGINE=1."""
    if _INSTALLED or os.environ.get("LOWBIT_W16_ENGINE", "0") != "1":
        return
    _install()


def _install() -> None:
    global _INSTALLED
    if os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"):
        # the plugin may run only its hybrid path: its base GEMM is quant_method.apply -- ours below -- and our adapter
        # kernels add the LoRA in place; its fused W8A8 GEMM and the hybridq path (CUTLASS on our codes) must stay unused
        if os.environ.get("LOWBIT_DISPATCH", "auto") != "hybrid" or os.environ.get("LOWBIT_HYBRIDQ", "0") not in ("0", ""):
            raise RuntimeError("w16_engine with the plugin needs LOWBIT_DISPATCH=hybrid and LOWBIT_HYBRIDQ=0 "
                               "(otherwise the plugin's W8A8 GEMMs would run)")
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="lowbit_w8a16_linear", op_func=_linear, mutates_args=[], fake_impl=_linear_fake)
    orig_process = Fp8PerTensorOnlineLinearMethod.process_weights_after_loading
    orig_apply = Fp8PerTensorOnlineLinearMethod.apply

    def process(self, layer):
        orig_process(self, layer)
        ok = (isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel) and layer.weight.dtype == torch.float8_e4m3fn
              and layer.weight_scale.numel() == 1 and layer.weight.t().is_contiguous())
        layer._lowbit_w16 = ok      # [K, N] view of the [N, K] codes vLLM quantized into (as CUTLASS reads them)
        stats["w16_layers" if ok else "fallback_layers"] += 1

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_lowbit_w16", False) or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.vllm.lowbit_w8a16_linear(x2, layer.weight.t(), layer.weight_scale.reshape(1))
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    Fp8PerTensorOnlineLinearMethod.process_weights_after_loading = process
    Fp8PerTensorOnlineLinearMethod.apply = apply
    _INSTALLED = True
    print("[lowbit] w16 engine installed: vLLM's online FP8 layers compute W8A16 through our kernel", flush=True)
