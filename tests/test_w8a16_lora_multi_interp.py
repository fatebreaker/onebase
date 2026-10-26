"""CPU check of the fused weight-only FP8 (W8A16) + multi-adapter LoRA kernel in Triton's interpreter (seconds; no GPU).

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_w8a16_lora_multi_interp.py
As tests/test_fp8_lora_multi_interp.py: random token -> adapter assignments (tokens without an adapter, adapters absent
from the batch), 1-3 slices, split-K, decode and prefill modes, bf16 and FP8-stored adapters, and a batch with no adapter.
The interpreter's bf16 tl.dot is wrong, so dots run in fp32 (DOT_F32).
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.kernels import fp8_lora as F1
from lowbit_lora.kernels import w8a16_lora_multi as W

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
CG = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}


def main() -> int:
    torch.manual_seed(0)
    worst = 0.0
    for K, slices, r, M, L, split, no_lora in [(64, (32, 16, 16), 16, 21, 4, 2, False), (96, (64,), 32, 5, 3, 2, False),
                                               (64, (32, 32), 16, 40, 5, 1, False), (64, (32,), 16, 9, 2, 2, True)]:
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a = (torch.randn(L, S * r, K) / K ** 0.5).to(torch.bfloat16)
        b = (torch.randn(L, N, r) * 0.05).to(torch.bfloat16)
        x = torch.randn(M, K).to(torch.bfloat16)
        tok = torch.full((M,), -1, dtype=torch.int32) if no_lora else torch.randint(-1, L - 1, (M,), dtype=torch.int32)
        meta = W.segments(tok, L)
        q = [F1.quantize_rows_fp8(a[i]) for i in range(L)]
        a8 = torch.stack([t for t, _ in q]); sa = torch.stack([s for _, s in q])
        q = [F1.quantize_rows_fp8(b[i]) for i in range(L)]
        b8 = torch.stack([t for t, _ in q]); sb = torch.stack([s for _, s in q])
        for fp8_ad in (False, True):
            kw = {"a_scale": sa, "b_scale": sb} if fp8_ad else {}
            aa, bb = (a8, b8) if fp8_ad else (a, b)
            op = W.MultiFusedW8A16LoRA(w_q, s_w, aa, bb, slices, 2.0, max_m=64, split_k=split, **kw)
            op.dot_f32 = True
            ref = W.reference_multi_w16(x, tok, w_q, s_w, aa, bb, slices, 2.0, **kw)
            yp = op(x, tok, meta, "prefill", cfg_s=CS, cfg_g=CG).float().clone()
            yd = op(x, tok, meta, "decode", cfg_s=CS, cfg_g=CG).float().clone()
            ep = float((yp - ref).norm() / ref.norm())
            ed = float((yd - ref).norm() / ref.norm())
            worst = max(worst, ep, ed)
            print(f"K={K} slices={slices} r={r} M={M} adapters={meta['n_active']}/{L} split={split} "
                  f"fp8_adapter={fp8_ad}: prefill {ep:.1e}  decode {ed:.1e}")
    for M, N, K in [(1, 96, 160), (37, 64, 128), (130, 128, 96)]:   # the base-only launcher (w16_engine's GEMM), every M bucket
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        x = torch.randn(M, K).to(torch.bfloat16)
        y = W.w8a16_linear(x, w_q, s_w, dot_f32=True).float()
        ref = x.float() @ (w_q.float() * s_w).t()
        e = float((y - ref).norm() / ref.norm())
        worst = max(worst, e)
        print(f"w8a16_linear M={M} N={N} K={K}: {e:.1e}")
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
