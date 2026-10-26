"""Hopper fused path (MLSys v2, docs/PLAN_MLSYS_V2.md): per projection, rank, batch size and number of distinct adapters,
CUDA-graph timings of
  ref      vLLM's quantize + CUTLASS FP8 GEMM, no LoRA
  stock    ref, then vLLM's lora_shrink / lora_expand
  hybrid   ref, then our shrink and expand-add (what the H100 dispatch keeps today)
  fused    our shrink, then our Triton GEMM with the LoRA epilogue over the adapter-sorted rows (fp8_lora_vllm)
  tma      our shrink, then the TMA-fed GEMM with the epilogue in natural order (fp8_lora_tma), default tiles and a search
with the output of each fused variant checked against stock (relative error below 2e-2).
    python bench/hopper_tma_bench.py [--model qwen3_8b] [--ranks 16,64] [--ms 1,16,32,64,128,256] [--ns 1,8] --out results/hopper_tma.jsonl
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


def tma_candidates(M: int, widths) -> list[dict]:
    out = []
    bms = (64,) if M <= 64 else (64, 128)
    for bm, bn, bk, st, ws in itertools.product(bms, (32, 64, 128, 256), (64, 128, 256), (3, 4, 5), (False, True)):
        if any(w % bn for w in widths) or bm * bn >= 128 * 256 and st > 3 or bk == 256 and (bn >= 256 or st > 4):
            continue      # (32-wide tiles: more thread blocks at decode, where 64-wide ones leave SMs idle)
        for nw in ((4,) if bm == 64 and bn <= 128 else (4, 8)):
            out.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": nw, "NUM_STAGES": st, "WS": ws, "GROUP_M": 8})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="1,16,32,64,128,256")
    ap.add_argument("--ns", default="1,8")
    ap.add_argument("--search", type=int, default=1, help="1: also search the TMA tiles")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels.fp8_lora_tma import pick_cfg_tma
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
                rec = {**info, "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M, "n": n,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                rec["t_ref"] = timer.graph_time(lambda: vendor(x))
                rec["t_stock"] = timer.graph_time(stock)
                op.use_tma = False
                rec["t_hybrid"] = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                rec["t_fused"] = timer.graph_time(lambda: op._run(x, margs))
                rec["err_fused"] = float((op._run(x, margs).float() - y_ref).norm() / y_ref.norm())
                op.use_tma = True
                op.tma_cfg = None
                try:
                    rec["err_tma"] = float((op._run(x, margs).float() - y_ref).norm() / y_ref.norm())
                    rec["t_tma"] = timer.graph_time(lambda: op._run(x, margs))
                except Exception as e:   # noqa: BLE001
                    rec["tma_error"] = repr(e)[:300]
                if a.search and "t_tma" in rec:
                    best = (rec["t_tma"], pick_cfg_tma(M, slices))
                    for cg in tma_candidates(M, slices):
                        op.tma_cfg = cg
                        try:
                            err = float((op._run(x, margs).float() - y_ref).norm() / y_ref.norm())
                            if err > 2e-2:
                                continue
                            t = timer.graph_time(lambda: op._run(x, margs))
                        except Exception:   # noqa: BLE001 -- shared memory, unsupported tile
                            continue
                        if t < best[0]:
                            best = (t, cg)
                    rec["t_tma_best"], rec["tma_best"] = best
                op.use_tma, op.tma_cfg = False, None
                for k in list(rec):
                    if k.startswith("t_") and isinstance(rec[k], float):
                        rec[k] = round(rec[k], 2)
                print(json.dumps({k: rec.get(k) for k in ("proj", "r", "M", "n", "t_ref", "t_stock", "t_hybrid", "t_fused", "t_tma",
                                                          "t_tma_best", "err_tma", "tma_error")}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
