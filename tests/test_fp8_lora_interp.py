"""CPU check of the fused FP8 + LoRA kernels' indexing in Triton's interpreter (seconds; no GPU).

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_fp8_lora_interp.py
The interpreter's bf16/fp8 tl.dot is wrong, so the kernels run with DOT_F32 (operands cast to fp32 before each dot);
x is quantized in-kernel, and the interpreter's fp32 -> e4m3 conversion may round differently from torch's, so the
reference uses the kernel's own x_q (prefill mode exposes it). What is checked: tiling, split-K partial sums, slice
selection per N-tile, masks for ragged M and K, and the epilogue, on shapes with 1-3 slices.
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.kernels import fp8_lora as F

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
CG = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}


def ref_from_xq(xq, sx, w_q, s_w, x, a, b, slices, scale, has_lora=True, a_scale=None, b_scale=None):
    if a_scale is not None:
        a = a.float() * a_scale[:, None]
    if b_scale is not None:
        b = b.float() * b_scale[:, None]
    y = (xq.float() * sx[:, None]) @ (w_q.float() * s_w).t()
    if has_lora:
        R = b.shape[1]
        h = (x.float() @ a.float().t() * scale).to(torch.bfloat16).float()
        off = 0
        for s, n in enumerate(slices):
            y[:, off:off + n] += h[:, s * R:(s + 1) * R] @ b[off:off + n].float().t()
            off += n
    return y


def main() -> int:
    torch.manual_seed(0)
    worst = 0.0
    for K, slices, r, M, split in [(64, (32, 16, 16), 16, 5, 2), (128, (64, 64), 16, 33, 3), (96, (64,), 32, 1, 2),
                                   (80, (32, 16, 16), 16, 17, 1)]:
        N = sum(slices)
        S = len(slices)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a = (torch.randn(S * r, K) / K ** 0.5).to(torch.bfloat16)
        b = (torch.randn(N, r) * 0.05).to(torch.bfloat16)
        x = torch.randn(M, K).to(torch.bfloat16)
        a8, sa = F.quantize_rows_fp8(a)
        b8, sb = F.quantize_rows_fp8(b)
        for has_lora, fp8_ad in ((True, False), (False, False), (True, True)):
            kw = {"a_scale": sa, "b_scale": sb} if fp8_ad else {}
            op = F.FusedFP8LoRA(w_q, s_w, a8 if fp8_ad else a, b8 if fp8_ad else b, slices, 2.0, max_m=64,
                                split_k=split, has_lora=has_lora, **kw)
            op.dot_f32 = True
            yp = op(x, "prefill", cfg_s=CS, cfg_g=CG).float().clone()
            xq, sx = op.xq[:M].clone(), op.sx[:M].clone()
            ref = ref_from_xq(xq, sx, w_q, s_w, x, a8 if fp8_ad else a, b8 if fp8_ad else b, slices, 2.0, has_lora,
                              **kw)
            yd = op(x, "decode", cfg_s=CS, cfg_g=CG).float().clone()       # quantizes x itself, split-K shrink
            ep = float((yp - ref).norm() / ref.norm())
            ed = float((yd - ref).norm() / ref.norm())
            worst = max(worst, ep, ed)
            print(f"K={K} slices={slices} r={r} M={M} split={split} lora={has_lora} fp8_adapter={fp8_ad}: "
                  f"prefill {ep:.1e}  decode {ed:.1e}")
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
