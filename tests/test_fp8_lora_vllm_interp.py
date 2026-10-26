"""CPU check of the vLLM-layout fused FP8 + LoRA kernels (fp8_lora_vllm.py) in Triton's interpreter, driven by vLLM's
own LoRAKernelMeta (on CPU) and vLLM's per-slice stacked buffers. Covers multi-adapter batches, tokens without an
adapter, and a no-adapter batch right after an adapter batch (vLLM then leaves token_lora_mapping stale).

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_fp8_lora_vllm_interp.py
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
    torch.manual_seed(0)
    worst = 0.0
    # r_chunk: the epilogue's rank chunk. 16 at rank 32 runs it in two chunks (narrow NVFP4 dequantization of B); None
    # at rank 128 is the hosts' rule: the whole rank for bf16/FP8 fused, 64-wide chunks (vectorized NVFP4
    # dequantization) for NVFP4-stored B and every expand-add
    for K, widths, R, L, split, r_chunk in [(64, (32, 16, 16), 16, 4, 2, 64), (96, (64,), 32, 3, 3, 64),
                                            (64, (32, 32), 16, 5, 1, 64), (96, (64,), 32, 3, 3, 16),
                                            (64, (32,), 128, 2, 2, None),
                                            (64, (32, 16, 16, 32), 16, 3, 2, 64)]:   # 4 slices (linear-attention in_proj)
        N, S = sum(widths), len(widths)
        w = torch.randn(N, K) * 0.05
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)                    # [L, S*R, K] for the reference
        B = torch.cat([t[:, 0] for t in b_st], dim=1)                    # [L, N, R]
        op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split)
        op.dot_f32 = True
        op8 = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split, lora_fp8=True)   # FP8-stored adapters
        op8.dot_f32 = True
        A8 = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op8.a8, op8.sa)], dim=1)   # dequantized
        B8 = torch.cat([q[:, 0].float() * sc[..., None] for q, sc in zip(op8.b8, op8.sb)], dim=1)
        op4 = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=64, split_k=split, lora_fp4=True)   # NVFP4-stored adapters
        op4.dot_f32 = True
        from lowbit_lora.kernels.nvfp4_lora_vllm import nvfp4_dequant_torch as dq4

        def deq4(q, qs, g, i):   # one adapter's [rows, cols] from the NVFP4 stacks
            return dq4(q[i, 0], qs[i, 0], 1.0) * g[i]
        A4 = torch.cat([torch.stack([deq4(q, qs, op4.ag[s], i) for i in range(L)]) for s, (q, qs) in
                        enumerate(zip(op4.a4, op4.a4s))], dim=1)
        B4 = torch.cat([torch.stack([deq4(q, qs, op4.bg[s], i) for i in range(L)]) for s, (q, qs) in
                        enumerate(zip(op4.b4, op4.b4s))], dim=1)
        rel4 = float((A4 - A).norm() / A.norm()), float((B4 - B).norm() / B.norm())
        op.r_chunk = op8.r_chunk = op4.r_chunk = r_chunk
        meta = LoRAKernelMeta.make(L, 64, device="cpu")
        for M, kind in [(21, "mixed"), (5, "mixed"), (13, "none"), (40, "single"), (24, "padded")]:
            x = torch.randn(M, K).to(torch.bfloat16)
            if kind == "mixed":
                tok = torch.randint(-1, L - 1, (M,), dtype=torch.int32)
            elif kind == "single":
                tok = torch.full((M,), L - 1, dtype=torch.int32)
            else:
                tok = torch.full((M,), -1, dtype=torch.int32)
            if kind == "padded":   # graph padding: metadata for the first M-5 tokens, stale entries beyond
                tok = torch.randint(-1, L - 1, (M,), dtype=torch.int32)
                tok[0] = 0                                     # at least one adapter token: sorted mode
                meta.token_indices_sorted_by_lora_ids[M - 5:M] = 10 ** 6
                meta.token_lora_mapping[M - 5:M] = L - 1         # a stale adapter id the padding must ignore
                meta.prepare_tensors(tok[: M - 5])
                tok[M - 5:] = -1                                 # expected: padding rows get the base GEMM only
            else:
                meta.prepare_tensors(tok)
            margs = meta.meta_args(M, False)
            for mode in ("prefill", "decode"):
                y = op(x, margs, mode=mode, cfg_s=CS, cfg_g=CG).float()
                if mode == "prefill":
                    xq = op.scratch.xq[: M * K].view(M, K).clone()
                    sx = op.scratch.sx[:M].clone()
                ref = FM.reference_multi(x, tok, w_q, s_w, A, B, widths, 1.0, xq=xq, sx=sx)
                err = float((y - ref).norm() / ref.norm())
                y8 = op8(x, margs, mode=mode, cfg_s=CS, cfg_g=CG).float()
                ref8 = FM.reference_multi(x, tok, w_q, s_w, A8, B8, widths, 1.0, xq=xq, sx=sx)
                err8 = float((y8 - ref8).norm() / ref8.norm())
                y4 = op4(x, margs, mode=mode, cfg_s=CS, cfg_g=CG).float()
                ref4 = FM.reference_multi(x, tok, w_q, s_w, A4, B4, widths, 1.0, xq=xq, sx=sx)
                err4 = float((y4 - ref4).norm() / ref4.norm())
                # hybrid: a vendor base GEMM (here the torch reference without adapters) + our LoRA kernels in place
                base_fn = lambda xx: FM.reference_multi(xx, torch.full((xx.shape[0],), -1, dtype=torch.int32), w_q, s_w,  # noqa: E731
                                                        A, B, widths, 1.0).to(torch.bfloat16)
                # hybridq: the vendor GEMM on OUR codes and scales (torch here, as CUTLASS would), then expand-add
                base_q_fn = lambda xq_, sx_: ((xq_.float() * sx_) @ (w_q.float() * s_w).t()).to(torch.bfloat16)  # noqa: E731
                errh = errq = 0.0
                for o, Ar, Br in ((op, A, B), (op8, A8, B8), (op4, A4, B4)):
                    o.base_fn = base_fn
                    yh = o._run(x, margs, mode=mode, cfg_s=CS, cfg_g=CG, hybrid=True).float()
                    rh = FM.reference_multi(x, tok, w_q, s_w, Ar, Br, widths, 1.0)
                    errh = max(errh, float((yh - rh).norm() / rh.norm()))
                    o.base_q_fn = base_q_fn
                    yq = o._run(x, margs, mode=mode, cfg_s=CS, cfg_g=CG, hybrid="q").float()
                    rq = FM.reference_multi(x, tok, w_q, s_w, Ar, Br, widths, 1.0,
                                            xq=o.scratch.xq[: M * K].view(M, K).clone(), sx=o.scratch.sx[:M].clone())
                    errq = max(errq, float((yq - rq).norm() / rq.norm()))
                    o.base_fn = o.base_q_fn = None
                worst = max(worst, err, err8, err4, errh, errq)
                print(f"K={K} widths={widths} R={R} L={L} split={split} chunk={r_chunk} M={M} {kind:6s} {mode:7s}: rel err {err:.1e}, "
                      f"FP8 adapters {err8:.1e}, NVFP4 adapters {err4:.1e}, hybrid (all formats) {errh:.1e}, "
                      f"hybridq {errq:.1e}")
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
