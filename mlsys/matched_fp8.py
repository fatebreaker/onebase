"""Matched-precision LoRA RL (lowbit-lora P2, 2026-10-11): make the trainer compute on the FP8 base its rollout engine
serves, so the sampler and the trainer are one policy up to kernel rounding.

vLLM's online FP8 (``--quantization fp8`` on a bf16 checkpoint; Fp8PerTensorOnlineLinearMethod) quantizes every
linear layer of the model except the LM head and the GDN conv1d:
  weights      one scale per *fused* tensor, s_W = amax(|W|) / 448 over the whole fused weight (q/k/v together,
               gate/up together, the GDN input projections qkv/z and b/a together; o, out and down alone);
               codes e4m3(W / s_W), round to nearest even, saturating;
  activations  dynamic per-token scales s_x = max(amax_row(|x|) / 448, 1 / (448 * 512)), codes e4m3(x / s_x);
  output       (x_codes @ W_codes^T) * s_x * s_W, fp32 accumulation, bf16 out.
The LoRA path (exact adapter, our plugin or vLLM's) reads the bf16 activations, so it stays bf16 here too.

``patch_model(model)`` rewrites the forward of each such nn.Linear in a Hugging Face model in place (they stay
nn.Linear, so PEFT still wraps them, and the LoRA delta is added by PEFT on bf16 inputs as in the engine). The base
weight is replaced by its FP8 codes and scale (half the memory of bf16). The forward uses torch._scaled_mm where the
GPU has FP8 tensor cores (sm_89+), else the same math in bf16. The backward passes the gradient straight through the
activation rounding to the (frozen) dequantized weight.

    from mlsys.matched_fp8 import patch_model; stats = patch_model(model)
"""
from __future__ import annotations

import re

import torch

FP8_MAX = 448.0
MIN_SCALE = 1.0 / (FP8_MAX * 512.0)

# HF module-name suffixes that vLLM fuses into one quantized tensor (one weight scale per group), for Qwen3/3.5,
# Llama and Mistral; any other nn.Linear outside SKIP is its own group.
GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"), ("in_proj_qkv", "in_proj_z"),
          ("in_proj_b", "in_proj_a"))
SKIP = re.compile(r"(^|\.)(lm_head|conv1d|embed_tokens|score)$")


def _quant_rows(x: torch.Tensor):
    """Per-token (row) dynamic FP8 quantization of a 2-D bf16 tensor, as vLLM's dynamic per-token kernel."""
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    s = torch.clamp(amax / FP8_MAX, min=MIN_SCALE)
    q = torch.clamp(x.float() / s, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


_STATE = {"scaled_mm_failed": None}         # set to the error text if torch._scaled_mm is unusable here


def _mm_codes(xq, sx, w_codes, w_scale, exact: bool):
    """(x_codes @ W_codes^T) * s_x * s_W without FP8 tensor cores. e4m3 codes are exact in bf16; exact=True also
    accumulates and scales in fp32 (tests); otherwise a bf16 matmul (fp32 accumulate) rounds the sum once more."""
    if exact:
        return ((xq.to(torch.float32) @ w_codes.to(torch.float32).t()) * sx * w_scale).to(torch.bfloat16)
    return ((xq.to(torch.bfloat16) @ w_codes.to(torch.bfloat16).t()).float() * sx * w_scale).to(torch.bfloat16)


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_codes, w_scale, use_scaled_mm: bool):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        xq, sx = _quant_rows(x2)
        y = None
        if use_scaled_mm and _STATE["scaled_mm_failed"] is None:
            n = w_codes.shape[0]
            try:
                y = torch._scaled_mm(xq, w_codes.t(), scale_a=sx,
                                     scale_b=w_scale.reshape(1, 1).expand(1, n).contiguous(), out_dtype=torch.bfloat16)
            except RuntimeError as e:               # unsupported shape/scale layout: fall back once, loudly
                _STATE["scaled_mm_failed"] = str(e)[:300]
                print(f"[matched_fp8] torch._scaled_mm unusable, bf16 fallback: {_STATE['scaled_mm_failed']}", flush=True)
        if y is None:
            y = _mm_codes(xq, sx, w_codes, w_scale, exact=not x.is_cuda)
        ctx.save_for_backward(w_codes, w_scale)
        return y.reshape(*shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, gy):
        w_codes, w_scale = ctx.saved_tensors
        w = (w_codes.to(torch.float32) * w_scale).to(torch.bfloat16)      # frozen base: straight-through on x
        gx = gy.reshape(-1, gy.shape[-1]).to(torch.bfloat16) @ w
        return gx.reshape(*gy.shape[:-1], w.shape[1]), None, None, None


def _make_forward(use_scaled_mm: bool, parts: str = "both"):
    """parts: "both" (vLLM's W8A8), "w" (FP8 weights, bf16 activations) or "a" (FP8 activations, bf16 weights) --
    the last two only for diagnosing which half of the emulation an engine mismatch comes from."""
    def forward(self, x):
        if parts == "both":
            y = _FP8Linear.apply(x, self.w_codes, self.w_scale, use_scaled_mm)
        elif parts == "w":
            y = x @ (self.w_codes.to(torch.float32) * self.w_scale).to(torch.bfloat16).t()
        else:
            xq, sx = _quant_rows(x.reshape(-1, x.shape[-1]))
            xd = (xq.to(torch.float32) * sx).to(torch.bfloat16).reshape(x.shape)
            xd = x + (xd - x).detach()                          # straight-through
            y = xd @ self.w_bf16.t()
        if self.bias is not None:
            y = y + self.bias
        return y
    return forward


def _group_key(name: str):
    parent, _, leaf = name.rpartition(".")
    for g in GROUPS:
        if leaf in g:
            return parent + "|" + "+".join(g)
    return name


def patch_model(model: torch.nn.Module, use_scaled_mm: bool | None = None, parts: str = "both") -> dict:
    """Quantize every eligible nn.Linear of `model` (before PEFT wraps it) to vLLM's online FP8, in place."""
    if use_scaled_mm is None:
        use_scaled_mm = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and not SKIP.search(n)}
    groups: dict[str, list[str]] = {}
    for n in lins:
        groups.setdefault(_group_key(n), []).append(n)
    fwd = _make_forward(use_scaled_mm, parts)
    n_params = 0
    for key, names in groups.items():
        with torch.no_grad():
            amax = max(float(lins[n].weight.detach().abs().max()) for n in names)
            scale = torch.tensor(amax / FP8_MAX, dtype=torch.float32, device=lins[names[0]].weight.device)
            for n in names:
                m = lins[n]
                w = m.weight.detach()
                codes = torch.clamp(w.float() / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                m.register_buffer("w_codes", codes.contiguous(), persistent=False)
                m.register_buffer("w_scale", scale.clone(), persistent=False)
                if parts == "a":
                    m.register_buffer("w_bf16", w.clone(), persistent=False)
                n_params += w.numel()
                # keep a 1-element placeholder weight: PEFT reads weight.dtype/device and in/out features only
                m.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
                m.forward = fwd.__get__(m, type(m))
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return {"linear_layers": len(lins), "scale_groups": len(groups), "quantized_params": n_params,
            "scaled_mm": use_scaled_mm, "parts": parts}
