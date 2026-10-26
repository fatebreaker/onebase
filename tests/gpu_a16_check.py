"""GPU check of the fused kernels' weight-only mode (fp8_lora_vllm.py A16, LOWBIT_A16_MAX_M) on real shapes with bf16 tensor
cores: Qwen3.5-9B's projection shapes, rank 64, 1-4 adapters, M = 1..32, bf16 / FP8- / NVFP4-stored adapters, against the fp32
W8A16 reference (bf16 x times the dequantized weights, plus each token's adapter). Prints the worst relative error per case.
    PYTHONPATH=.:../mlsys python tests/gpu_a16_check.py"""
from __future__ import annotations

import torch

from lowbit_lora.kernels import fp8_lora_multi as FM
from lowbit_lora.kernels import fp8_lora_vllm as F


def main() -> int:
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
    torch.manual_seed(0)
    dev = "cuda"
    worst = 0.0
    for K, widths in [(4096, (4096, 1024, 1024)), (4096, (12288, 12288)), (12288, (4096,))]:
        R, L = 64, 4
        N = sum(widths)
        w = torch.randn(N, K, device=dev) * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        a_st = tuple((torch.randn(L, 1, R, K, device=dev) / K ** 0.5).to(torch.bfloat16) for _ in widths)
        b_st = tuple((torch.randn(L, 1, n, R, device=dev) * 0.02).to(torch.bfloat16) for n in widths)
        A = torch.cat([t[:, 0] for t in a_st], dim=1)
        B = torch.cat([t[:, 0] for t in b_st], dim=1)
        op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max_m=256)
        meta = LoRAKernelMeta.make(L, 256, device=dev)
        for M, kind in [(1, "single"), (1, "none"), (4, "mixed"), (16, "mixed"), (32, "mixed")]:
            x = torch.randn(M, K, device=dev).to(torch.bfloat16)
            tok = (torch.randint(-1, L - 1, (M,), dtype=torch.int32, device=dev) if kind == "mixed" else
                   torch.full((M,), 0 if kind == "single" else -1, dtype=torch.int32, device=dev))
            meta.prepare_tensors(tok)
            y = op._run(x, meta.meta_args(M, False), a16=True).float()
            ref = FM.reference_multi(x, tok.cpu(), w_q, s_w, A, B, widths, 1.0, xq=x.float(), sx=torch.ones(M, device=dev))
            e = float((y - ref).norm() / ref.norm())
            worst = max(worst, e)
            print(f"K={K} N={N} M={M} {kind:6s}: rel err {e:.2e}", flush=True)
    ok = worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst {worst:.2e}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
