"""CPU check of the vLLM-layout fused kernels' weight-only mode (fp8_lora_vllm.py, A16 / LOWBIT_A16_MAX_M) in Triton's
interpreter: y = x @ (W_q * s_w)^T + LoRA with bf16 activations (no activation rounding), for bf16, FP8- and NVFP4-stored
adapters, multi-adapter batches, tokens without an adapter and a no-adapter batch; reference fp8_lora_multi.reference_multi
given x itself as the codes with unit scales (so its base GEMM is W8A16: bf16 x times the dequantized weights). Small shapes, so it runs in about a minute on a login node.

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_a16_vllm_interp.py
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.kernels import fp8_lora_multi as FM
from lowbit_lora.kernels import fp8_lora_vllm as F

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
CG = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}


def main() -> int:
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch as dq4
    torch.manual_seed(0)
    worst = 0.0
    for K, widths, R, L, split in [(64, (32, 16, 16), 16, 4, 2), (96, (64,), 32, 3, 1)]:
        N, S = sum(widths), len(widths)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)
        B = torch.cat([t[:, 0] for t in b_st], dim=1)
        ops = {"bf16": F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split),
               "fp8": F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split, lora_fp8=True),
               "fp4": F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split, lora_fp4=True)}
        o8, o4 = ops["fp8"], ops["fp4"]
        refs = {"bf16": (A, B),
                "fp8": (torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(o8.a8, o8.sa)], dim=1),
                        torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(o8.b8, o8.sb)], dim=1)),
                "fp4": (torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * o4.ag[s][i] for i in range(L)])
                                   for s, (q, qs) in enumerate(zip(o4.a4, o4.a4s))], dim=1),
                        torch.cat([torch.stack([dq4(q[i, 0], qs[i, 0], 1.0) * o4.bg[s][i] for i in range(L)])
                                   for s, (q, qs) in enumerate(zip(o4.b4, o4.b4s))], dim=1))}
        for o in ops.values():
            o.dot_f32 = True
        meta = LoRAKernelMeta.make(L, 64, device="cpu")
        for M, kind in [(21, "mixed"), (5, "mixed"), (13, "none"), (32, "single")]:
            x = torch.randn(M, K).to(torch.bfloat16)
            tok = (torch.randint(-1, L - 1, (M,), dtype=torch.int32) if kind == "mixed" else
                   torch.full((M,), L - 1 if kind == "single" else -1, dtype=torch.int32))
            meta.prepare_tensors(tok)
            margs = meta.meta_args(M, False)
            one = torch.ones(M)
            errs = {}
            for name, o in ops.items():
                y = o._run(x, margs, cfg_s=CS, cfg_g=CG, a16=True).float()
                ref = FM.reference_multi(x, tok, w_q, s_w, *refs[name], widths, 1.0, xq=x.float(), sx=one)   # W8A16
                errs[name] = float((y - ref).norm() / ref.norm())
            # the switch: __call__ takes the weight-only path for M <= a16_max_m
            o = ops["bf16"]
            o.a16_max_m = 64
            yc = o(x, margs, cfg_s=CS, cfg_g=CG).float()
            o.a16_max_m = 0
            rc = FM.reference_multi(x, tok, w_q, s_w, A, B, widths, 1.0, xq=x.float(), sx=one)
            errs["call"] = float((yc - rc).norm() / rc.norm())
            worst = max(worst, *errs.values())
            print(f"K={K} widths={widths} R={R} L={L} split={split} M={M} {kind:6s}: " +
                  ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
