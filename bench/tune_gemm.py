"""Tile search for the fused FP8 GEMM with the LoRA epilogue (GPU, ~20 min): per projection and batch size, time
fp8_lora_vllm's fused path (shrink at its tuned default, then our FP8 GEMM + expand epilogue) over candidate GEMM tiles,
against the stock path (vLLM's quantize + CUTLASS, then lora_shrink / lora_expand) and the plugin's hybrid path. The
GEMM tiles of pick_cfg were tuned on the L40S; on the H100 they leave our decode GEMM up to 2x behind CUTLASS for the
4096-wide projections (job 65486898), so the timed dispatch falls back to the hybrid path there.
    python bench/tune_gemm.py [--model llama31_8b] [--ranks 16,64] [--ms 1,16,32,64,128,256] --out results/gemm_tune.jsonl
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


def candidates(M: int, widths) -> list[dict]:
    """~20-70 tiles per batch size: decode (M <= 64) is memory-bound, so narrow N tiles (more thread blocks) and deep
    pipelines; larger M gets wider tiles and 8 warps."""
    if M <= 64:
        bms, bns, stages = [bm for bm in (16, 32, 64) if bm <= max(16, M)], (32, 64, 128), (3, 4, 5)
    else:
        bms, bns, stages = [bm for bm in (32, 64, 128) if bm <= M], (64, 128, 256), (3, 4)
    out = []
    for bm, bn, bk, st in itertools.product(bms, bns, (128, 256), stages):
        if any(x % bn for x in widths) or (bm * bn * bk >= 128 * 256 * 128 and st > 3):
            continue
        for w in ((4,) if bm <= 32 else (4, 8)):
            out.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": st, "GROUP_M": 8})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="1,16,32,64,128,256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels import fp8_lora_vllm as F
    torch.manual_seed(0)
    timer = Timer(warm=5, reps=30)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", "")}
    ms = [int(m) for m in a.ms.split(",")]
    L = 4
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
            for M in ms:
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                tok = torch.zeros(M, dtype=torch.int32, device="cuda")
                meta.prepare_tensors(tok)
                margs = meta.meta_args(M, False)
                hb = hbuf[: S * M * r].view(S, M, r)

                def stock():
                    y = vendor(x)
                    lora_shrink(x, a_st, hb, *margs, 1.0)
                    lora_expand(hb, b_st, y, *margs, offset_start=0, add_inputs=True)
                    return y
                t_ref = timer.graph_time(lambda: vendor(x))
                t_stock = timer.graph_time(stock)
                t_hybrid = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                y_ref = stock().float()
                _, cg0 = F.pick_cfg(M, N, K, r, slices)
                res = []
                for cg in [dict(cg0, default=True)] + candidates(M, slices):
                    cfg = {k: v for k, v in cg.items() if k != "default"}
                    try:
                        y = op._run(x, margs, cfg_g=cfg).float()
                        if float((y - y_ref).norm() / y_ref.norm()) > 2e-2:
                            continue
                        res.append((timer.graph_time(lambda: op._run(x, margs, cfg_g=cfg)), cg))
                    except Exception:   # noqa: BLE001 -- out of shared memory, unsupported tile and the like
                        continue
                res.sort(key=lambda t: t[0])
                rec = {**info, "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M,
                       "t_ref": round(t_ref, 2), "t_stock": round(t_stock, 2), "t_hybrid": round(t_hybrid, 2),
                       "t_default": next((round(t_, 2) for t_, c in res if c.get("default")), None),
                       "t_best": round(res[0][0], 2), "best": res[0][1],
                       "top5": [(round(t, 2), c) for t, c in res[:5]], "n_cand": len(res),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                print(json.dumps({k: rec[k] for k in ("proj", "r", "M", "t_ref", "t_stock", "t_hybrid", "t_default",
                                                      "t_best", "best")}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
