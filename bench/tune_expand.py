"""Tile search for the hybrid path's expand-add launch (GPU, minutes): with the vendor GEMM computing the base, our
GEMM kernel runs with BASE=False and an empty main loop (y += LoRA in place), yet it is launched with the GEMM's tiles
(pick_cfg), whose software pipeline sizes the kernel's shared memory statically whether the loop runs or not. On the
Blackwell slice the NVFP4 op's LoRA part then costs ~2x the FP8 op's (job 65488272: qkv r64 M 256, 81 vs 41 us).
Times the hybrid path (vendor GEMM + our shrink + expand-add) over candidate expand tiles, per projection and M, one
adapter, against the stock path; parity against the default configuration.
    python bench/tune_expand.py --op fp8|nvfp4 [--model llama31_8b] [--ranks 16,64] [--ms 64,128,256,1024]
        --out results/expand_tune.jsonl
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


def candidates(widths, op: str) -> list[dict]:
    out = []
    for bm, bn, bk, w, st in itertools.product((16, 32, 64, 128), (64, 128, 256), (64, 128), (4, 8), (1, 2)):
        if any(x % bn for x in widths) or (bm >= 128 and w == 4) or (bm <= 32 and w == 8):
            continue
        out.append({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": st, "GROUP_M": 8})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", choices=("fp8", "nvfp4"), required=True)
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="64,128,256,1024")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
    torch.manual_seed(0)
    timer = Timer(warm=5, reps=30)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", ""), "op": a.op}
    ms = [int(m) for m in a.ms.split(",")]
    L = 4
    meta = LoRAKernelMeta.make(L, max(ms), device="cuda")
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        if a.op == "fp8":
            from lowbit_lora.kernels import fp8_lora_vllm as F
            s_w = (w.abs().max() / 448.0).reshape(1).float()
            w_q = (w / s_w).to(torch.float8_e4m3fn)
            w_t = w_q.t()

            def vendor(xx):
                xq, sx = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
        else:
            from lowbit_lora.kernels import nvfp4_lora_vllm as F
            wb = w.to(torch.bfloat16)
            w_global = F.global_scale_for(wb.float())
            wg_t = torch.tensor([w_global], dtype=torch.float32, device="cuda")
            w_q_sw, w_sf_sw = ops.scaled_fp4_quant(wb, wg_t)
            w_q, w_sf = ops.scaled_fp4_quant(wb, wg_t, is_sf_swizzled_layout=False)
        for r in [int(v) for v in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            hbuf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            op = None
            for M in ms:
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                if a.op == "nvfp4":
                    x_global = F.global_scale_for(x.float())
                    xg_t = torch.tensor([x_global], dtype=torch.float32, device="cuda")
                    alpha = torch.tensor([1.0 / (x_global * w_global)], dtype=torch.float32, device="cuda")

                    def vendor(xx, _xg=xg_t, _al=alpha):
                        xq, xs = ops.scaled_fp4_quant(xx, _xg)
                        return ops.cutlass_scaled_fp4_mm(xq, w_q_sw, xs, w_sf_sw, _al, torch.bfloat16)
                    if op is None:
                        op = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, x_global, a_st, b_st, max(ms))
                    op.x_global.copy_(xg_t)
                    op.alpha.copy_(alpha)
                    run = lambda cg=None: op(x, margs, cfg_g=cg, base_fn=vendor, hybrid_m=0)   # noqa: E731
                else:
                    if op is None:
                        op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms))
                        op.base_fn = vendor
                    run = lambda cg=None: op._run(x, margs, cfg_g=cg, hybrid=True)   # noqa: E731
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
                y0 = run().float()
                t_default = timer.graph_time(run)
                res = []
                for cg in candidates(slices, a.op):
                    try:
                        y = run(cg).float()
                        if float((y - y0).norm() / y0.norm()) > 1e-3:
                            continue
                        res.append((timer.graph_time(lambda: run(cg)), cg))
                    except Exception:   # noqa: BLE001
                        continue
                res.sort(key=lambda t: t[0])
                rec = {**info, "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M,
                       "t_ref": round(t_ref, 2), "t_stock": round(t_stock, 2), "t_default": round(t_default, 2),
                       "t_best": round(res[0][0], 2) if res else None, "best": res[0][1] if res else None,
                       "top5": [(round(t, 2), c) for t, c in res[:5]], "n_cand": len(res),
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                print(json.dumps({k: rec[k] for k in ("proj", "r", "M", "t_ref", "t_stock", "t_default", "t_best",
                                                      "best")}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
