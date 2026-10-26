"""GEMM-BENCH (descriptive; RL-KERNELS): the trainer's base GEMMs at Qwen3.5-9B's projection shapes and a trainer micro-batch's
M, in bf16 (cuBLAS) against FP8 through torch._scaled_mm with row-wise scales (the matched trainer's: per-token activations,
per-tensor weight) and with tensor-wise scales, forward (x @ W^T) and backward (g @ W, which reads W's codes transposed),
plus the per-call transpose of the codes and the K1 per-row quantization. CUDA-event timing, median of --reps.
    python bench/gemm_bench.py [--m 3120] [--out results/gemm_bench.jsonl]"""
from __future__ import annotations

import argparse
import json

import torch


def bench(fn, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=3120)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from lowbit_lora.fp8_train import quant_rows_triton
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    M = a.m
    for K, N in ((4096, 12288), (12288, 4096), (4096, 4096), (4096, 1024), (4096, 8192)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        g = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq = (w.float() / 0.001).clamp(-448, 448).to(torch.float8_e4m3fn)
        wtq = wq.t().contiguous()
        xq, sx = quant_rows_triton(x)
        gq, sg = quant_rows_triton(g)
        one = torch.ones((), device=dev)
        r = {"gpu": gpu, "M": M, "K": K, "N": N}
        r["bf16_fwd_ms"] = bench(lambda: x @ w.t(), a.reps)
        r["bf16_bwd_ms"] = bench(lambda: g @ w, a.reps)
        r["fp8_row_fwd_ms"] = bench(lambda: torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=one.reshape(1, 1).expand(1, N).contiguous(),
                                                             out_dtype=torch.bfloat16), a.reps)
        r["fp8_row_bwd_ms"] = bench(lambda: torch._scaled_mm(gq, wtq.t(), scale_a=sg, scale_b=one.reshape(1, 1).expand(1, K).contiguous(),
                                                             out_dtype=torch.bfloat16), a.reps)
        r["fp8_tensor_fwd_ms"] = bench(lambda: torch._scaled_mm(xq, wq.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16), a.reps)
        from lowbit_lora.train_lora import k2_plain
        sx1, sg1 = sx.reshape(-1).contiguous(), sg.reshape(-1).contiguous()
        r["k2_fwd_ms"] = bench(lambda: k2_plain(xq, sx1, wq, one), a.reps)
        r["k2_bwd_ms"] = bench(lambda: k2_plain(gq, sg1, wtq, one), a.reps)
        from lowbit_lora.kernels.fp8_train_kernels import gemm as tgemm
        r["tk_fwd_ms"] = bench(lambda: tgemm(xq, sx1, wq, one), a.reps)
        r["tk_bwd_inplace_ms"] = bench(lambda: tgemm(gq, sg1, wq, one, transposed=True), a.reps)
        r["transpose_ms"] = bench(lambda: wq.t().contiguous(), a.reps)
        r["quant_x_ms"] = bench(lambda: quant_rows_triton(x), a.reps)
        fl = 2 * M * K * N
        for k in ("bf16_fwd", "bf16_bwd", "fp8_row_fwd", "fp8_row_bwd", "fp8_tensor_fwd", "k2_fwd", "k2_bwd", "tk_fwd", "tk_bwd_inplace"):
            r[k + "_tflops"] = round(fl / (r[k + "_ms"] * 1e-3) / 1e12, 1)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
