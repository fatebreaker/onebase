"""The matched-precision engine with the swap-AB W8A16 GEMM on Hopper (MATCHED-HOPPER-KERNEL, docs/REGISTRY.md): as
lowbit_lora/w16_engine.py (vLLM's online FP8 layers compute W8A16 from the engine's own e4m3 codes, in-process), but on sm_90
the GEMM is kernels/w8a16_swap.w8a16_linear_swap (the widened codes as wgmma's register operand, tokens as its N), with tiles
from the H100 sweep (results/w16_swap.jsonl) per batch size and projection shape; other GPUs keep w16_engine's kernel. Same
arithmetic: bf16 products of the widened codes, fp32 accumulation, the per-tensor scale once, one bf16 rounding.

    from lowbit_lora import w16_engine_swap; w16_engine_swap.install()   # before vllm.LLM(...), engine in this process
w16_engine.py is left unchanged (RL-MATCHED runs it); this module registers its own custom op, so a compiled graph names the
kernel it calls (and vLLM's compile cache must still be fresh per engine: MISMATCH-DECOMP-2-CACHE).
"""
from __future__ import annotations

import os

import torch

_INSTALLED = False
stats = {"w16_layers": 0, "fallback_layers": 0, "swap": False}
SWAP_CFG: dict = {}   # (M bucket, N, K) -> config; filled from the H100 sweep before any end-to-end data


def _bucket(m: int) -> int:
    for b in (1, 8, 16, 32, 64, 128, 256, 512):
        if m <= b:
            return b
    return 512


def load_cfg(path: str, gpu_substr: str = "H100", job: str | None = None) -> int:
    """Fill SWAP_CFG from bench/w16_swap_bench.py rows (the best swept tiles per (M bucket, N, K) on that GPU; the newest job
    unless one is named). Shapes the sweep did not cover take the tiles of the nearest swept shape (by N*K) at that bucket."""
    import json
    rows = [json.loads(x) for x in open(path).read().splitlines() if x.strip()]
    rows = [r for r in rows if gpu_substr in r.get("gpu", "") and r.get("swap_best_cfg")]
    if not rows:
        return 0
    job = job or max(r["job"] for r in rows)
    for r in rows:
        if r["job"] == job:
            SWAP_CFG[(_bucket(r["M"]), r["N"], r["K"])] = dict(r["swap_best_cfg"])
    return len(SWAP_CFG)


def _lookup(m: int, n: int, k: int) -> dict | None:
    b = _bucket(m)
    c = SWAP_CFG.get((b, n, k))
    if c is not None or not SWAP_CFG:
        return c
    import math
    near = [(abs(math.log(nn * kk) - math.log(n * k)), key) for key in SWAP_CFG for (bb, nn, kk) in [key] if bb == b]
    return SWAP_CFG[min(near)[1]] if near else None


def _linear(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    """x [M, K] bf16, w [N, K] e4m3 codes (contiguous), sw [1] fp32 -> [M, N] bf16 (w16_engine.py's convention)."""
    if stats["swap"]:
        from lowbit_lora.kernels.w8a16_swap import swap_cfg, w8a16_linear_swap
        m, k = x.shape
        n = w.shape[0]
        return w8a16_linear_swap(x, w, sw, _lookup(m, n, k) or swap_cfg(m, n, k))
    from lowbit_lora.kernels.w8a16_lora_multi import w8a16_linear
    return w8a16_linear(x, w, sw)


def _linear_fake(x: torch.Tensor, w: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return torch.empty(x.shape[0], w.shape[0], dtype=torch.bfloat16, device=x.device)


def install(force_swap: bool | None = None) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") != "0":
        raise RuntimeError("w16_engine_swap patches this process: set VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine runs in it")
    if os.environ.get("LOWBIT_FUSED", "0") not in ("0", "", "false", "False"):
        raise RuntimeError("w16_engine_swap runs with the plugin off (LOWBIT_FUSED=0)")
    major, _ = torch.cuda.get_device_capability()
    stats["swap"] = (major == 9) if force_swap is None else force_swap
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFP8ScaledMMLinearKernel
    from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(op_name="lowbit_w8a16s_linear", op_func=_linear, mutates_args=[], fake_impl=_linear_fake)
    orig_process = Fp8PerTensorOnlineLinearMethod.process_weights_after_loading
    orig_apply = Fp8PerTensorOnlineLinearMethod.apply

    def process(self, layer):
        orig_process(self, layer)
        ok = (isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel) and layer.weight.dtype == torch.float8_e4m3fn
              and layer.weight_scale.numel() == 1 and layer.weight.t().is_contiguous())
        layer._lowbit_w16s = ok     # [K, N] view of the [N, K] codes vLLM quantized into (as CUTLASS reads them)
        stats["w16_layers" if ok else "fallback_layers"] += 1

    def apply(self, layer, x, bias=None):
        if not getattr(layer, "_lowbit_w16s", False) or x.dtype != torch.bfloat16:
            return orig_apply(self, layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y = torch.ops.vllm.lowbit_w8a16s_linear(x2, layer.weight.t(), layer.weight_scale.reshape(1))
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])

    Fp8PerTensorOnlineLinearMethod.process_weights_after_loading = process
    Fp8PerTensorOnlineLinearMethod.apply = apply
    _INSTALLED = True
    print(f"[lowbit] w16 engine (swap-AB on sm_90: {stats['swap']}) installed: vLLM's online FP8 layers compute W8A16", flush=True)
