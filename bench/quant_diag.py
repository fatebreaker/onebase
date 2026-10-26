"""Diagnostic (GPU): how vLLM's per-token dynamic FP8 quantization (scaled_fp8_quant) compares, element by element,
with exact recipes and with the fused kernels' own quantization, on real-scale random activations.

    python bench/quant_diag.py
Prints, for each recipe, the share of elements whose FP8 code differs from vLLM's and the scale mismatch.
"""
from __future__ import annotations

import torch


def main() -> int:
    from vllm import _custom_ops as ops

    from lowbit_lora.kernels.fp8_lora import FusedFP8LoRA
    torch.manual_seed(0)
    M, K = 512, 4096
    x = (torch.randn(M, K, device="cuda") * torch.rand(M, 1, device="cuda") * 4).to(torch.bfloat16)
    x[:, :8] *= 20                                                         # a few outlier channels, as in LLMs
    q_v, s_v = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    s_v = s_v.view(-1)
    xf = x.float()
    amax = xf.abs().amax(1)
    floor = 1.0 / (448.0 * 512.0)
    recipes = {
        "div, scale amax/448": (amax / 448.0).clamp_min(floor),
        "mul by 1/scale": None,
    }
    s_ref = (amax / 448.0).clamp_min(floor)
    s_mul = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).clamp_min(floor)
    print(f"scale = amax * fp32(1/448): identical to vLLM's in {int((s_v == s_mul).sum())}/{M} rows")
    print(f"scale: max rel diff vLLM vs amax/448 = {float(((s_v - s_ref).abs() / s_ref).max()):.3e}; "
          f"identical scales {int((s_v == s_ref).sum())}/{M}")
    q_div = (xf / s_v[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q_mul = (xf * (1.0 / s_v)[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q_ref = (xf / s_ref[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    for name, q in (("x / s_vllm (torch RN)", q_div), ("x * (1/s_vllm) (torch RN)", q_mul), ("x / (amax/448) (torch RN)", q_ref)):
        d = (q.view(torch.uint8) != q_v.view(torch.uint8)).float().mean()
        print(f"{name:30s}: codes differing from vLLM {100 * float(d):.4f}%")
    # the fused kernel's own x_q (prefill mode writes it)
    N = 1024
    w_q = (torch.randn(N, K, device="cuda") * 0.02 / (0.1 / 448)).clamp(-448, 448).to(torch.float8_e4m3fn)
    s_w = torch.tensor([0.1 / 448], device="cuda")
    a = torch.zeros(16, K, device="cuda", dtype=torch.bfloat16)
    b = torch.zeros(N, 16, device="cuda", dtype=torch.bfloat16)
    op = FusedFP8LoRA(w_q, s_w, a, b, (N,), 1.0, M)
    op(x, "prefill")
    q_k, s_k = op.xq[:M], op.sx[:M]
    print(f"fused kernel scale vs vLLM: max rel diff {float(((s_k - s_v).abs() / s_v).max()):.3e}")
    d = (q_k.view(torch.uint8) != q_v.view(torch.uint8)).float().mean()
    print(f"fused kernel x_q: codes differing from vLLM {100 * float(d):.4f}%")
    d2 = (q_k.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean()
    print(f"fused kernel x_q: codes differing from exact x/(amax/448) {100 * float(d2):.4f}%")
    # GEMM outputs, no adapter: our two kernels vs vLLM's quant + cutlass_scaled_mm, bit for bit
    y_v = ops.cutlass_scaled_mm(q_v, w_q.t(), s_v.view(-1, 1), s_w, torch.bfloat16)
    plain = FusedFP8LoRA(w_q, s_w, a, b, (N,), 1.0, M, has_lora=False)
    for mode in ("prefill", "decode"):
        xs = x if mode == "prefill" else x[:64]
        yk = plain(xs, mode).clone()
        yv = y_v[: xs.shape[0]]
        diff = (yk.view(torch.int16) != yv.view(torch.int16)).float().mean()
        rel = float((yk.float() - yv.float()).norm() / yv.float().norm())
        print(f"no-adapter GEMM ({mode}, M={xs.shape[0]}): bf16 outputs differing from vLLM {100 * float(diff):.4f}%, rel {rel:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
