"""Rounding schemes applied to (base weight W, RL update dW): what a deployed / synced model actually receives.

    base_q, rl_q = apply_scheme(W, dW, scheme)       # float32 tensors; delivered update = rl_q - base_q

W is the stored base weight (bf16 values held in float32); dW is the true update in float32 (for LoRA,
scale * B @ A). Scheme names (used by census.py, vllm_ext.py and the sync-mode trainer):

    bf16_merge       ON-GRID   bf16(W + dW), base W itself                      (verl merge path; any bf16 merge)
    bf16_sr_merge    ON-GRID   stochastic-rounding bf16 merge                   (unbiased fix)
    fp16_merge       ON-GRID   fp16(W + dW): bf16 values are exact in fp16, 8x finer spacing
    <fmt>_shared     OFF-GRID  Q(W + dW) and Q(W) on ONE grid derived from W      (pure dither: sparse +-step flips)
    <fmt>_fresh      OFF-GRID  Q(W + dW) and Q(W), each with its own scales     (what quantization tools do: the
                               moved group max rescales the whole group, adding a drift term to the flips)
    <fmt>_frozen     ON-GRID   Q(Q(W) + dW) on the base's scales                (merge into an already-quantized base)
    <fmt>_requant    ~ON-GRID  Q(Q(W) + dW) with scales recomputed from Q(W) + dW (PEFT's merge into a bitsandbytes
                               4-bit base: dequantize, add, re-quantize; small updates leave the block maxima, and so
                               the grid, unchanged)
    <fmt>_jitter<k>  OFF-GRID  shared, the grid's scales multiplied by 1 + U[0, 0.05) with seed k
    unmerged_<fmt>   EXACT     Q(W) + dW (adapter kept in activation space on a quantized base)

    <fmt> in: fp8, int8, int4g128, nvfp4, mxfp4, nf4 (bitsandbytes NF4, blocks of 64; fresh and requant only)
                                                    (unmerged_ also accepts bf16)
    e8m<k>_frozen    ON-GRID   a float grid with k stored mantissa bits (bf16: k = 7): base_q = R_k(W),
                               rl_q = R_k(base_q + dW); k < 7 sets the dead zone's depth, rho grows as 2^(7-k)
"""
from __future__ import annotations

import torch

from . import quant

FORMATS = ("fp8", "int8", "int4g128", "nvfp4", "mxfp4", "nf4")


def lora_delta(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """PEFT convention: A [r, in], B [out, r]; delta = scale * B @ A with scale = lora_alpha / r."""
    return scale * (B.float() @ A.float())


def _fresh(x: torch.Tensor, fmt: str, gen: torch.Generator | None = None, jitter: float = 0.0
           ) -> tuple[torch.Tensor, torch.Tensor | None]:
    if fmt == "fp8":
        return quant.fp8_quant(x)
    if fmt.startswith("int"):
        bits, group = quant.parse_int(fmt)
        return quant.int_quant(x, bits, group, gen=gen, jitter=jitter)
    if fmt == "nvfp4":
        return quant.nvfp4_quant(x), None
    if fmt == "mxfp4":
        return quant.mxfp4_quant(x), None
    if fmt == "nf4":
        return quant.nf4_quant(x), None
    if fmt == "bf16":
        return quant.bf16(x), None
    raise ValueError(f"unknown format {fmt!r}")


def _on_scale(x: torch.Tensor, fmt: str, scale: torch.Tensor, mode: str = "nearest",
              gen: torch.Generator | None = None) -> torch.Tensor:
    if fmt == "fp8":
        return quant.fp8_quant(x, scale=scale, mode=mode, gen=gen)[0]
    if fmt.startswith("int"):
        bits, group = quant.parse_int(fmt)
        return quant.int_quant(x, bits, group, scale=scale, mode=mode, gen=gen)[0]
    raise ValueError(f"frozen grid needs explicit scales; {fmt!r} has none here")


def apply_scheme(W: torch.Tensor, dW: torch.Tensor, scheme: str, seed: int = 0
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    W, dW = W.float(), dW.float()
    gen = torch.Generator(device=W.device).manual_seed(seed)
    if scheme.startswith("e8m") and scheme.endswith("_frozen"):
        k = int(scheme[3:].split("_")[0])
        base_q = quant.e8m(W, k)
        return base_q, quant.e8m(base_q + dW, k)
    if scheme == "bf16_merge":
        return W, quant.bf16(W + dW)
    if scheme == "bf16_sr_merge":
        return W, quant.bf16(W + dW, mode="stochastic", gen=gen)
    if scheme == "fp16_merge":
        return W, quant.fp16(W + dW)
    if scheme.startswith("unmerged_"):
        base_q = _fresh(W, scheme.removeprefix("unmerged_"))[0]
        return base_q, base_q + dW
    fmt, _, mode = scheme.rpartition("_")
    if mode == "fresh":
        return _fresh(W, fmt)[0], _fresh(W + dW, fmt)[0]
    if mode == "requant":                                 # PEFT's merge into a quantized base
        base_q = _fresh(W, fmt)[0]
        return base_q, _fresh(base_q + dW, fmt)[0]
    if mode == "shared":
        base_q, scale = _fresh(W, fmt)
        if scale is None:
            raise ValueError(f"shared grid needs a scaled format, not {fmt!r}")
        return base_q, _on_scale(W + dW, fmt, scale)
    if mode == "frozen":
        if fmt in ("nvfp4", "mxfp4"):                    # native FP4 checkpoint: block scales stay fixed
            base_q, scale, _ = quant.fp4_quant_with_scale(W, fmt)
            return base_q, quant.fp4_on_scale(base_q + dW, fmt, scale)
        base_q, scale = _fresh(W, fmt)
        return base_q, _on_scale(base_q + dW, fmt, scale)
    if mode == "srfrozen":                                # the frozen grid, stochastically rounded (SR merge)
        if fmt in ("nvfp4", "mxfp4"):
            base_q, scale, _ = quant.fp4_quant_with_scale(W, fmt)
            return base_q, quant.fp4_on_scale(base_q + dW, fmt, scale, mode="stochastic", gen=gen)
        base_q, scale = _fresh(W, fmt)
        return base_q, _on_scale(base_q + dW, fmt, scale, mode="stochastic", gen=gen)
    if mode.startswith("jitter"):
        k = int(mode.removeprefix("jitter"))
        g = torch.Generator(device=W.device).manual_seed(1000 + k)
        base_q, scale = _fresh(W, fmt, gen=g, jitter=0.05)
        if scale is None:
            raise ValueError(f"jitter needs a scaled format, not {fmt!r}")
        return base_q, _on_scale(W + dW, fmt, scale)       # one jittered grid for both: positions re-drawn
    raise ValueError(f"unknown scheme {scheme!r}")


def is_on_grid(scheme: str) -> bool:
    return scheme in ("bf16_merge", "bf16_sr_merge", "fp16_merge") or scheme.endswith(("_frozen", "_srfrozen"))
