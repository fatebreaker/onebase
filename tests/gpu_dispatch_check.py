"""GPU check of the timed dispatch and its "hybrids" candidate (the vendor GEMM, then vLLM's own LoRA kernels on
vLLM's bf16 adapters), for the FP8 op and, on Blackwell, the NVFP4 op (seconds to minutes):
  - parity of every path (fused, hybrid, hybrids) against an fp32 reference at M 1 / 64 / 256, one and four adapters;
  - the dispatch table tune_dispatch() picks per batch-size bucket (Llama-3.1-8B qkv and down shapes, rank 16 / 64);
  - warmup() runs (it compiles every path the table may pick).
    python tests/gpu_dispatch_check.py
"""
from __future__ import annotations

import torch

from lowbit_lora.kernels import fp8_lora_multi as FM
from lowbit_lora.kernels import fp8_lora_vllm as F


def main() -> int:
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta
    torch.manual_seed(0)
    cc = torch.cuda.get_device_capability()
    print(torch.cuda.get_device_name(0), cc, flush=True)
    L, bad = 4, 0
    meta = LoRAKernelMeta.make(L, 512, device="cuda")
    for K, slices in ((4096, (4096, 1024, 1024)), (14336, (4096,))):
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()
        for r in (16, 64):
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))

            def vendor(xx):
                xq, sx = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
            op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, 512)
            op.base_fn = vendor
            for M in (1, 64, 256):
                for n in (1, 4):
                    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                    tok = (torch.arange(M, device="cuda") % n).to(torch.int32)
                    meta.prepare_tensors(tok)
                    margs = meta.meta_args(M, False)
                    ref = FM.reference_multi(x, tok, w_q, s_w, A, B, slices, 1.0)
                    errs = {h: float((op._run(x, margs, hybrid=h).float() - ref).norm() / ref.norm())
                            for h in (False, True, "s")}
                    ok = all(e <= 2e-2 for e in errs.values())
                    bad += not ok
                    print(f"fp8 K={K} r={r} M={M} n={n}: " + "  ".join(f"{k}: {v:.1e}" for k, v in errs.items())
                          + ("" if ok else "  WRONG"), flush=True)
            op.warmup()
            print(f"fp8 K={K} r={r} dispatch: {op.dispatch}", flush=True)
    if cc[0] >= 10:   # the NVFP4 op (Blackwell)
        from lowbit_lora.kernels import nvfp4_lora_vllm as F4
        for K, slices in ((4096, (4096, 1024, 1024)), (14336, (4096,))):
            N, S = sum(slices), len(slices)
            w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
            w_global = F4.global_scale_for(w.float())
            wg_t = torch.tensor([w_global], dtype=torch.float32, device="cuda")
            w_q_sw, w_sf_sw = ops.scaled_fp4_quant(w, wg_t)
            w_q, w_sf = ops.scaled_fp4_quant(w, wg_t, is_sf_swizzled_layout=False)
            for r in (16, 64):
                A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
                B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
                a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
                offs = [sum(slices[:s]) for s in range(S)]
                b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous()
                             for s in range(S))
                xg = 448.0
                xg_t = torch.tensor([xg], dtype=torch.float32, device="cuda")
                alpha = torch.tensor([1.0 / (xg * w_global)], dtype=torch.float32, device="cuda")

                def vendor4(xx):
                    xq, xs = ops.scaled_fp4_quant(xx, xg_t)
                    return ops.cutlass_scaled_fp4_mm(xq, w_q_sw, xs, w_sf_sw, alpha, torch.bfloat16)
                op = F4.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, xg, a_st, b_st, 512)
                op.base_fn = vendor4
                for M in (1, 64, 256):
                    for n in (1, 4):
                        x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                        tok = (torch.arange(M, device="cuda") % n).to(torch.int32)
                        meta.prepare_tensors(tok)
                        margs = meta.meta_args(M, False)
                        y_f = op(x, margs, hybrid_m=1 << 30, use_dispatch=False).float()
                        ref = F4.reference(x, tok, w_q, w_sf, w_global, xg, A, B, slices,
                                           xq=op.xq[:M].clone(), xsf=op.sxq[:M].clone())
                        outs = {"fused": y_f, "hybrid": op(x, margs, hybrid_m=0, use_dispatch=False).float(),
                                "hybrids": op._run_stock(x, margs).float()}
                        errs = {k: float((v - ref).norm() / ref.norm()) for k, v in outs.items()}
                        ok = all(e <= 2e-2 for e in errs.values())
                        bad += not ok
                        print(f"nvfp4 K={K} r={r} M={M} n={n}: " + "  ".join(f"{k}: {v:.1e}" for k, v in errs.items())
                              + ("" if ok else "  WRONG"), flush=True)
                op.warmup()
                print(f"nvfp4 K={K} r={r} dispatch: {op.dispatch}", flush=True)
    print("PASS" if not bad else f"FAIL {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
