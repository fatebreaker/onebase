"""CPU check of the multi-adapter fused FP8 + LoRA kernels in Triton's interpreter (seconds; no GPU).

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_fp8_lora_multi_interp.py
Random token -> adapter assignments (including tokens without an adapter and adapters absent from the batch), 1-3
slices, split-K, decode and prefill modes, bf16 and FP8-stored adapters. The interpreter's bf16/fp8 tl.dot is wrong,
so dots run in fp32 (DOT_F32); the reference uses the kernel's own x_q (exposed in prefill mode).
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.kernels import fp8_lora as F1
from lowbit_lora.kernels import fp8_lora_multi as F

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
CG = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}


def main() -> int:
    torch.manual_seed(0)
    worst = 0.0
    for K, slices, r, M, L, split in [(64, (32, 16, 16), 16, 21, 4, 2), (96, (64,), 32, 5, 3, 2),
                                      (64, (32, 32), 16, 40, 5, 1)]:
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a = (torch.randn(L, S * r, K) / K ** 0.5).to(torch.bfloat16)
        b = (torch.randn(L, N, r) * 0.05).to(torch.bfloat16)
        x = torch.randn(M, K).to(torch.bfloat16)
        tok = torch.randint(-1, L - 1, (M,), dtype=torch.int32)          # slot L-1 never used: absent adapters
        meta = F.segments(tok, L)
        q = [F1.quantize_rows_fp8(a[i]) for i in range(L)]
        a8 = torch.stack([t for t, _ in q]); sa = torch.stack([s for _, s in q])
        q = [F1.quantize_rows_fp8(b[i]) for i in range(L)]
        b8 = torch.stack([t for t, _ in q]); sb = torch.stack([s for _, s in q])
        for fp8_ad in (False, True):
            kw = {"a_scale": sa, "b_scale": sb} if fp8_ad else {}
            aa, bb = (a8, b8) if fp8_ad else (a, b)
            op = F.MultiFusedFP8LoRA(w_q, s_w, aa, bb, slices, 2.0, max_m=64, split_k=split, **kw)
            op.dot_f32 = True
            yp = op(x, tok, meta, "prefill", cfg_s=CS, cfg_g=CG).float().clone()
            ref = F.reference_multi(x, tok, w_q, s_w, aa, bb, slices, 2.0, xq=op.xq[:M].clone(),
                                    sx=op.sx[:M].clone(), **kw)
            yd = op(x, tok, meta, "decode", cfg_s=CS, cfg_g=CG).float().clone()
            ep = float((yp - ref).norm() / ref.norm())
            ed = float((yd - ref).norm() / ref.norm())
            worst = max(worst, ep, ed)
            print(f"K={K} slices={slices} r={r} M={M} adapters={meta['n_active']}/{L} split={split} "
                  f"fp8_adapter={fp8_ad}: prefill {ep:.1e}  decode {ed:.1e}")
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
