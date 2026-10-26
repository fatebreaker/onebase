"""Tile search for the LoRA shrink (GPU, minutes): time the plugin's hybrid path (vLLM's quantize + CUTLASS, then our
shrink and the expand-add epilogue) with candidate shrink tiles, per projection and batch size, one adapter. The H100
benchmark (job 65486898) showed 16-row shrink tiles re-reading A once per 16 tokens, which dominates at M >= 256 on
Hopper. --fmt fp8|fp4 searches with FP8- or NVFP4-stored adapters (their dequantization is repeated once per M tile,
job 65487248), checked against the same op's default configuration instead of the bf16 stock path. --op nvfp4
searches the NVFP4 op's shrink on Blackwell (vendor FP4 GEMM, then our shrink and expand-add). t_shipped times exactly
what the plugin runs (common.shrink_choice tiles and split).
    python bench/tune_shrink.py [--model llama31_8b] [--ranks 16,64] [--ms 64,128,256,512,1024,4096] [--fmt bf16] \
        --out results/shrink_tune.jsonl
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="64,128,256,512,1024,4096")
    ap.add_argument("--fmt", default="bf16", choices=("bf16", "fp8", "fp4"))
    ap.add_argument("--op", default="fp8", choices=("fp8", "nvfp4"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels import nvfp4_lora_vllm as F4
    torch.manual_seed(0)
    timer = Timer(warm=5, reps=30)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", ""), "fmt": a.fmt, "op": a.op}
    ms = [int(m) for m in a.ms.split(",")]
    L = 4
    meta = LoRAKernelMeta.make(L, max(ms), device="cuda")
    cands = [(dict(BLOCK_M=bm, BLOCK_SR=64, BLOCK_K=bk, num_warps=4, num_stages=st), sp)
             for bm, bk, st, sp in itertools.product((16, 32, 64), (128, 256), (2, 3), (1, 2, 4, 8))]
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()
        if a.op == "nvfp4":
            wb = w.to(torch.bfloat16)
            w_global = F4.global_scale_for(wb.float())
            wg_t = torch.tensor([w_global], dtype=torch.float32, device="cuda")
            w4_sw, w4sf_sw = ops.scaled_fp4_quant(wb, wg_t)
            w4, w4sf = ops.scaled_fp4_quant(wb, wg_t, is_sf_swizzled_layout=False)
        for r in [int(v) for v in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            if a.op == "nvfp4":
                xg = 448.0 * 6 / 6.0                       # a fixed activation global (inputs ~N(0, 1), amax < 6)
                xg_t = torch.tensor([xg], dtype=torch.float32, device="cuda")
                alpha = torch.tensor([1.0 / (xg * w_global)], dtype=torch.float32, device="cuda")
                op = F4.VllmFusedNVFP4LoRA(w4, w4sf, w_global, xg, a_st, b_st, max(ms),
                                           lora_fmt={"bf16": "bf16", "fp8": "fp8", "fp4": "nvfp4"}[a.fmt])

                def vendor(xx):
                    xq_, xs_ = ops.scaled_fp4_quant(xx, xg_t)
                    return ops.cutlass_scaled_fp4_mm(xq_, w4_sw, xs_, w4sf_sw, alpha, torch.bfloat16)

                def run(cs=None, sp=None):
                    return op(x, margs, cfg_s=cs, base_fn=vendor, hybrid_m=0, split=sp)
            else:
                op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms), lora_fp8=a.fmt == "fp8",
                                        lora_fp4=a.fmt == "fp4")

                def vendor(xx):
                    xq_, sx_ = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                    return ops.cutlass_scaled_mm(xq_, w_t, sx_, s_w, torch.bfloat16)
                op.base_fn = vendor

                def run(cs=None, sp=None):
                    return op._run(x, margs, cfg_s=cs, hybrid=True, split=sp)
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
                cs0, cg = F.pick_cfg(M, N, K, r, slices)
                # bf16 adapters: the stock path; stored FP8/NVFP4 adapters: the op's own shipped path (same weights)
                y_ref = (stock() if a.fmt == "bf16" else run()).float()
                t_shipped = timer.graph_time(lambda: run())
                res = []
                for cs, sp in [(cs0, None)] + cands:
                    cs = dict(cs, BLOCK_SR=min(cs["BLOCK_SR"], r))
                    # split-K above SPLIT_M_MAX tokens: any split the op's shared scratch holds
                    hp_n = op.hp.numel() if a.op == "nvfp4" else op.scratch.hp.numel()
                    too_big = bool(sp) and sp > 1 and (sp * S * M * r > hp_n or (
                        a.op != "nvfp4" and sp * M > op.scratch.pmax.numel()))
                    if r % cs["BLOCK_SR"] or cs["BLOCK_M"] > max(16, 2 * M) or too_big:
                        continue
                    try:
                        y = run(cs, sp).float()
                        if float((y - y_ref).norm() / y_ref.norm()) > 2e-2:
                            continue
                        t_ = timer.graph_time(lambda: run(cs, sp))
                        res.append((t_, dict(cs, split=sp)))
                    except Exception:   # noqa: BLE001 -- out of shared memory and the like
                        continue
                res.sort(key=lambda t: t[0])
                rec = {**info, "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M, "t_ref": round(t_ref, 2),
                       "t_stock": round(t_stock, 2), "t_shipped": round(t_shipped, 2),
                       "t_default": next((round(t_, 2) for t_, c in res if c.get("split") is None), None),
                       "t_best": round(res[0][0], 2), "best": res[0][1], "top3": [(round(t, 2), c) for t, c in res[:3]],
                       # every candidate, compact: [time, BLOCK_M, BLOCK_SR, BLOCK_K, stages, split (0 = rule's)]
                       "all": [[round(t, 2), c["BLOCK_M"], c["BLOCK_SR"], c["BLOCK_K"], c["num_stages"], c["split"] or 0]
                               for t, c in res],
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                print(json.dumps({k: rec[k] for k in ("proj", "r", "M", "t_ref", "t_stock", "t_shipped", "t_best",
                                                      "best")}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
