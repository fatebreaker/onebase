"""CPU check of the fused NVFP4 + LoRA kernels (nvfp4_lora_vllm.py) in Triton's interpreter, driven by vLLM's own
LoRAKernelMeta (on CPU) and per-slice stacked buffers. The interpreter has no dot_scaled and no inline PTX, so this
checks the quantizer (codes and scales against the torch recipe), the packing, the tile/segment logic and the LoRA
epilogue; tests on sm_120 check dot_scaled and the PTX conversions (bench/nvfp4_check.py).

    TRITON_INTERPRET=1 PYTHONPATH=.:../mlsys python tests/test_nvfp4_lora_vllm_interp.py
"""
from __future__ import annotations

import os

import torch

from lowbit_lora.kernels import nvfp4_lora_vllm as F

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
CS = {"BLOCK_M": 16, "BLOCK_SR": 16, "BLOCK_K": 64, "num_warps": 4, "num_stages": 1}
CG = {"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 2, "num_warps": 4, "num_stages": 1}


def main() -> int:
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    torch.manual_seed(0)
    worst, worst_code = 0.0, 0.0
    # r_chunk: the epilogue's rank chunk. 16 at rank 32 runs it in two chunks (narrow NVFP4 dequantization of B); None
    # at rank 128 is the hosts' rule: the whole rank for bf16/FP8 fused, 64-wide chunks (vectorized NVFP4
    # dequantization) for NVFP4-stored B and every expand-add
    for K, widths, R, L, split, r_chunk in [(128, (32, 16, 16), 16, 4, 2, 64), (192, (64,), 32, 3, 3, 64),
                                            (128, (32, 32), 16, 5, 1, 64), (192, (64,), 32, 3, 3, 16),
                                            (128, (32,), 128, 2, 2, None),
                                            (128, (32, 16, 16, 32), 16, 3, 2, 64)]:   # 4 slices (linear-attention in_proj)
        N, S = sum(widths), len(widths)
        w = torch.randn(N, K) * 0.05
        w_global = F.global_scale_for(w)
        w_q, w_sf = F.nvfp4_quant_torch(w, w_global)
        a_st = tuple((torch.randn(L, 1, R, K) / K ** 0.5).to(torch.bfloat16) for _ in range(S))
        b_st = tuple((torch.randn(L, 1, n, R) * 0.05).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)
        B = torch.cat([t[:, 0] for t in b_st], dim=1)
        op = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, 1.0, a_st, b_st, max_m=96, split_k=split)
        op.dot_f32 = True
        op.r_chunk = r_chunk
        ops_lp = {}                                   # FP8- and NVFP4-stored adapters (adapter_store.AdapterStore)
        for fmt in ("fp8", "nvfp4"):
            o = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, 1.0, a_st, b_st, max_m=96, split_k=split, lora_fmt=fmt)
            o.dot_f32 = True
            o.r_chunk = r_chunk
            ops_lp[fmt] = (o, *o.store.dequantized())
        meta = LoRAKernelMeta.make(L, 96, device="cpu")
        for M, kind in [(21, "mixed"), (5, "mixed"), (13, "none"), (40, "single"), (24, "padded"), (80, "mixed")]:
            x = torch.randn(M, K).to(torch.bfloat16)
            x[:, :3] *= 8                                                # outlier channels
            x_global = F.global_scale_for(x)
            for o in [op] + [v[0] for v in ops_lp.values()]:
                o.x_global.fill_(x_global)
                o.alpha.fill_(1.0 / (x_global * w_global))
            if kind == "mixed":
                tok = torch.randint(-1, L - 1, (M,), dtype=torch.int32)
            elif kind == "single":
                tok = torch.full((M,), L - 1, dtype=torch.int32)
            else:
                tok = torch.full((M,), -1, dtype=torch.int32)
            if kind == "padded":   # CUDA-graph padding: metadata for the first M-5 tokens, stale entries beyond
                tok = torch.randint(-1, L - 1, (M,), dtype=torch.int32)
                tok[0] = 0
                meta.token_indices_sorted_by_lora_ids[M - 5:M] = 10 ** 6
                meta.token_lora_mapping[M - 5:M] = L - 1
                meta.prepare_tensors(tok[: M - 5])
                tok[M - 5:] = -1
            else:
                meta.prepare_tensors(tok)
            margs = meta.meta_args(M, False)
            y = op(x, margs, cfg_s=CS, cfg_g=CG).float()
            # the quantizer: codes and scales against the torch recipe (exact reciprocals in both)
            q_ref, sf_ref = F.nvfp4_quant_torch(x, x_global)
            dq = float((op.xq[:M] != q_ref).float().mean())
            ds = float((op.sxq[:M].view(torch.uint8) != sf_ref.view(torch.uint8)).float().mean())
            worst_code = max(worst_code, dq, ds)
            ref = F.reference(x, tok, w_q, w_sf, w_global, x_global, A, B, widths)
            err = float((y - ref).norm() / ref.norm())
            # hybrid: a vendor base GEMM (here the torch reference without adapters) + our LoRA added in place
            base_fn = lambda xx: F.reference(xx, torch.full((xx.shape[0],), -1), w_q, w_sf, w_global, x_global,  # noqa: E731
                                             A, B, widths).to(torch.bfloat16)
            yh = op(x, margs, cfg_s=CS, cfg_g=CG, base_fn=base_fn, hybrid_m=0).float()
            errh = float((yh - ref).norm() / ref.norm())
            worst = max(worst, err, errh)
            errs_lp = {}
            for fmt, (o, A_d, B_d) in ops_lp.items():      # low-precision adapters, fused and hybrid
                ref_lp = F.reference(x, tok, w_q, w_sf, w_global, x_global, A_d, B_d, widths)
                e1 = float((o(x, margs, cfg_s=CS, cfg_g=CG).float() - ref_lp).norm() / ref_lp.norm())
                e2 = float((o(x, margs, cfg_s=CS, cfg_g=CG, base_fn=base_fn, hybrid_m=0).float() - ref_lp).norm()
                           / ref_lp.norm())
                errs_lp[fmt] = max(e1, e2)
                worst = max(worst, e1, e2)
            print(f"K={K} widths={widths} R={R} L={L} split={split} chunk={r_chunk} M={M} {kind:6s}: rel err {err:.1e} hybrid {errh:.1e} "
                  f"fp8-ad {errs_lp['fp8']:.1e} nvfp4-ad {errs_lp['nvfp4']:.1e}  "
                  f"codes differing {100 * dq:.3f}%  scales differing {100 * ds:.3f}%")
    ok = worst < 1e-2 and worst_code < 1e-3
    print("PASS" if ok else "FAIL", f"worst rel err {worst:.2e}, worst code/scale mismatch {worst_code:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
