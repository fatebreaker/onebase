"""HOPPER-SWAP-K2 (docs/REGISTRY.md): the swap-AB K2 (kernels/fp8_lora_swap.py via lowbit_lora/hopper_swap.run_swap) against
the plugin's other Hopper paths and vLLM's, per projection of Qwen3-8B, rank, batch size M and number of distinct adapters n,
CUDA-graph timings (bench/kernel_bench.Timer, L2 flushed) of
  ref      vLLM's quantize + CUTLASS FP8 GEMM, no LoRA
  stock    ref, then vLLM's lora_shrink / lora_expand (default configs)
  hybrid   ref, then our shrink and expand-add (what the H100 dispatch keeps today)
  fused    our shrink, then our Triton GEMM with the LoRA epilogue over the adapter-sorted rows
  tma      our shrink, then the TMA-fed GEMM (fp8_lora_tma, default tiles)
  swap     our shrink, then the swap-AB GEMM: default tiles, and with --tune the best of a sweep (rank 64)
each fused variant's output checked against stock (relative error below 2e-2; failing configurations are skipped).
    python bench/swap_k2_bench.py [--ranks 16,64] [--ms 1,4,8,16,32,64,128] [--ns 1,8] [--tune] --out results/swap_k2.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info  # noqa: E402


def swap_candidates(M: int, widths) -> list[dict]:
    bms = (16,) if M <= 16 else (16, 32) if M <= 32 else (32, 64) if M <= 64 else (64, 128)
    sks = (1, 2, 4, 8) if M <= 64 else (1, 2)
    out = []
    for bm, bn, bk, sk, st in itertools.product(bms, (64, 128, 256), (128, 256), sks, (3, 4, 5)):
        if any(w % bn for w in widths) or (bn == 256 and bk == 256) or st * (bn * bk + bm * bk) > 200 * 1024:
            continue
        for nw in ((4,) if bn <= 128 else (4, 8)):
            out.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw, "num_stages": st, "GROUP_N": 8})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="1,4,8,16,32,64,128")
    ap.add_argument("--ns", default="1,8")
    ap.add_argument("--tune", action="store_true", help="sweep the swap tiles (rank 64 only)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.hopper_swap import run_swap
    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels.fp8_lora_swap import swap_k2_cfg
    torch.manual_seed(0)
    timer = Timer(warm=5, reps=30)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", ""), "kernel_sha": os.environ.get("LOWBIT_SHA", "")}
    ms = [int(m) for m in a.ms.split(",")]
    ns = [int(n) for n in a.ns.split(",")]
    L = max(8, max(ns))
    meta = LoRAKernelMeta.make(L, max(ms), device="cuda")
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()
        for r in [int(v) for v in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms))

            def vendor(xx):
                xq, sx = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
            op.base_fn = vendor
            hbuf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            for M, n in itertools.product(ms, ns):
                if n > M:
                    continue
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                tok = (torch.arange(M, device="cuda") % n).to(torch.int32)   # n distinct adapters, interleaved (decode)
                meta.prepare_tensors(tok)
                margs = meta.meta_args(M, False)
                hb = hbuf[: S * M * r].view(S, M, r)

                def stock():
                    y = vendor(x)
                    hb.zero_()
                    lora_shrink(x, a_st, hb, *margs, 1.0)
                    lora_expand(hb, b_st, y, *margs, offset_start=0, add_inputs=True)
                    return y
                y_ref = stock().float()
                err = lambda y: float((y.float() - y_ref).norm() / y_ref.norm())  # noqa: E731
                rec = {**info, "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M, "n": n,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                rec["t_ref"] = timer.graph_time(lambda: vendor(x))
                rec["t_stock"] = timer.graph_time(stock)
                op.use_tma = False
                rec["t_hybrid"] = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                rec["t_fused"] = timer.graph_time(lambda: op._run(x, margs))
                op.use_tma, op.tma_cfg = True, None
                try:
                    rec["err_tma"] = err(op._run(x, margs))
                    rec["t_tma"] = timer.graph_time(lambda: op._run(x, margs))
                except Exception as e:   # noqa: BLE001
                    rec["tma_error"] = repr(e)[:300]
                op.use_tma = False
                d = swap_k2_cfg(M, slices)
                rec["swap_default"] = d
                rec["err_swap"] = err(run_swap(op, x, margs, cfg=d))
                rec["t_swap"] = timer.graph_time(lambda: run_swap(op, x, margs, cfg=d))
                if a.tune and r == 64:
                    best = (rec["t_swap"], d, rec["err_swap"])
                    for cg in swap_candidates(M, slices):
                        try:
                            e = err(run_swap(op, x, margs, cfg=cg))
                            if e > 2e-2:
                                continue
                            t = timer.graph_time(lambda: run_swap(op, x, margs, cfg=cg))
                        except Exception:   # noqa: BLE001 -- shared memory, unsupported tile
                            continue
                        if t < best[0]:
                            best = (t, cg, e)
                    rec["t_swap_best"], rec["swap_best"], rec["err_swap_best"] = best
                for k in list(rec):
                    if k.startswith("t_") and isinstance(rec[k], float):
                        rec[k] = round(rec[k], 2)
                print(json.dumps({k: rec.get(k) for k in ("proj", "r", "M", "n", "t_ref", "t_stock", "t_hybrid", "t_fused", "t_tma",
                                                          "t_swap", "t_swap_best", "err_swap")}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
