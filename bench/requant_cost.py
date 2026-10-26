"""Cost of refreshing low-precision adapter copies when vLLM writes an adapter slot (GPU, seconds): the plugin
requantizes the slot's A and B to FP8 (per-row scales) or NVFP4 (E4M3 block scales of 16, a global per adapter and
slice) for every LoRA layer. Times one slot's refresh over a whole model (Llama-3.1-8B: 32 layers x 4 fused
projections), against copying the same bf16 adapter weights into the slots (what vLLM itself does per load).
    python bench/requant_cost.py [--ranks 16,64,128] [--out results/requant_cost.jsonl (default; "" = none)]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, gpu_info  # noqa: E402

LAYERS = 32


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--ranks", default="16,64,128")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--out", default="results/requant_cost.jsonl")
    a = ap.parse_args()
    from lowbit_lora.kernels import fp8_lora_vllm as F
    torch.manual_seed(0)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", "")}
    for r in [int(v) for v in a.ranks.split(",")]:
        ops = {"fp8": [], "fp4": []}
        srcs = []
        for proj, (K, slices) in SHAPES[a.model].items():
            N, S = sum(slices), len(slices)
            w_q = torch.zeros(N, K, device="cuda").to(torch.float8_e4m3fn)
            s_w = torch.ones(1, device="cuda")
            a_st = tuple((torch.randn(a.slots, 1, r, K, device="cuda") / K ** 0.5).to(torch.bfloat16) for _ in range(S))
            b_st = tuple((torch.randn(a.slots, 1, w, r, device="cuda") * 0.02).to(torch.bfloat16) for w in slices)
            srcs.append((a_st, b_st))
            for fmt in ops:
                ops[fmt].append(F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, 64, lora_fp8=fmt == "fp8",
                                                   lora_fp4=fmt == "fp4"))
        res = {}
        for fmt, lst in ops.items():
            for _ in range(3):   # warm-up (allocator, kernels)
                for op in lst:
                    op.requantize(1)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(LAYERS):   # every layer has these four projections; one slot each
                for op in lst:
                    op.requantize(1)
            torch.cuda.synchronize()
            res[fmt] = (time.perf_counter() - t) * 1e3
        # the bf16 slot copy vLLM does on an adapter load (device to device here: a lower bound on its own cost)
        dst = [tuple(torch.empty_like(t[1]) for t in a_st + b_st) for a_st, b_st in srcs]
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(LAYERS):
            for (a_st, b_st), d in zip(srcs, dst):
                for s_, d_ in zip(a_st + b_st, d):
                    d_.copy_(s_[1])
        torch.cuda.synchronize()
        res["bf16_copy"] = (time.perf_counter() - t) * 1e3
        mb = sum(t[1].numel() * 2 for a_st, b_st in srcs for t in a_st + b_st) * LAYERS / 1e6
        rec = {**info, "model": a.model, "r": r, "adapter_MB_bf16": round(mb, 1),
               "ms_per_slot": {k: round(v, 2) for k, v in res.items()},
               "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
