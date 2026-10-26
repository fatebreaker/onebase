"""NVFP4 (W4A4) fused LoRA kernels on sm_120 (Phase 1): parity with vLLM's FP4 ops, then timing against the stock path.

    python bench/nvfp4_check.py --out results/nvfp4_kernel.jsonl [--model qwen3_8b] [--ranks 16,64] [--ms ...] [--ns ...]
Per (projection, r, M, n):
  parity  q_codes / q_scales: share of x codes and block scales of our all-token pass that differ from vLLM's
          scaled_fp4_quant (linear layout); gemm_diff: share of bf16 outputs of our no-adapter GEMM that differ from
          cutlass_scaled_fp4_mm on vLLM's own codes, and its relative error; rel_err: the fused output against an fp32
          reference on our codes (nvfp4_lora_vllm.reference).
  timing  as bench/kernel_bench.py (CUDA graph, median of 200 replays, L2 flushed):
    ref    scaled_fp4_quant + cutlass_scaled_fp4_mm                    (no adapter: the merged cost)
    stock  ref + vLLM lora_shrink + lora_expand with n adapters (Punica metadata)
    base   our two kernels on a batch without adapters
    fused  our two kernels, n adapters
Tokens: round-robin over n adapters for M <= 64 (decode), contiguous runs for larger M (prefill chunks).
--lora-fmt fp8|nvfp4: the adapters are stored in that format (lowbit_lora.kernels.adapter_store), and the reference
uses the stored (dequantized) adapters; for M > 64 the hybrid path (vendor GEMM, then our shrink and expand-add) is
checked and timed as well ("hybrid").
--tune-base: instead, search GEMM tile configs for the no-adapter batch per (projection, M) against CUTLASS and write
the best per M to --out (results/nvfp4_tune.jsonl), for pick_cfg.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info, rel_err  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="1,16,64,256,1024")
    ap.add_argument("--ns", default="1,4,16,32")
    ap.add_argument("--pool", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lora-fmt", default="bf16", choices=("bf16", "fp8", "nvfp4"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any path is off its reference by > 2e-2")
    ap.add_argument("--tune-base", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels import nvfp4_lora_vllm as F
    cc = torch.cuda.get_device_capability()
    assert cc[0] * 10 + cc[1] >= 120 or cc[0] == 10, f"needs Blackwell, got sm_{cc[0]}{cc[1]}"
    print("cutlass fp4 supported:", ops.cutlass_scaled_mm_supports_fp4(cc[0] * 10 + cc[1]), flush=True)
    torch.manual_seed(0)
    timer = Timer(warm=5 if a.quick else 20, reps=20 if a.quick else 200)
    src = Path(F.__file__)   # the imported copy (jobs run a snapshot taken at job start, jobs/lib.sh)
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", ""),
            "kernel_sha": hashlib.sha256(src.read_bytes()).hexdigest()[:12], "pool": a.pool, "lora_fmt": a.lora_fmt}
    ms = [int(m) for m in a.ms.split(",")]
    L = a.pool
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_v = LoRAKernelMeta.make(L, max(ms), device="cuda")
    bad = 0
    if a.tune_base:
        return tune_base(a, F, ops, timer, info, meta_v, out)
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
        w_global = F.global_scale_for(w.float())
        wg_t = torch.tensor([w_global], dtype=torch.float32, device="cuda")
        w_q_sw, w_sf_sw = ops.scaled_fp4_quant(w, wg_t)                                   # CUTLASS layout
        w_q, w_sf = ops.scaled_fp4_quant(w, wg_t, is_sf_swizzled_layout=False)            # ours: linear scales
        assert torch.equal(w_q, w_q_sw)
        for r in [int(v) for v in a.ranks.split(",")]:
            A = (torch.randn(L, S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
            B = (torch.randn(L, N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(A[:, s * r:(s + 1) * r].reshape(L, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(B[:, offs[s]:offs[s] + slices[s]].reshape(L, 1, slices[s], r).contiguous() for s in range(S))
            buf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            op = None
            for M in ms:
                x = (torch.randn(M, K, device="cuda") * torch.rand(M, 1, device="cuda") * 2).to(torch.bfloat16)
                x[:, :8] *= 10                                                             # outlier channels
                x_global = F.global_scale_for(x.float())
                xg_t = torch.tensor([x_global], dtype=torch.float32, device="cuda")
                alpha = torch.tensor([1.0 / (x_global * w_global)], dtype=torch.float32, device="cuda")
                if op is None:
                    op = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, x_global, a_st, b_st, max(ms), lora_fmt=a.lora_fmt)
                    A_ref, B_ref = op.store.dequantized() if a.lora_fmt != "bf16" else (A, B)
                op.x_global.copy_(xg_t)
                op.alpha.copy_(alpha)
                for n in [int(v) for v in a.ns.split(",")]:
                    if n > L or n > M:
                        continue
                    ar = torch.arange(M, device="cuda")
                    tok = (ar % n if M <= 64 else (ar * n) // M).to(torch.int32)
                    hbuf = buf[: S * M * r].view(S, M, r)
                    none = torch.full((M,), -1, dtype=torch.int32, device="cuda")
                    meta_v.prepare_tensors(none)
                    margs0 = tuple(t.clone() if torch.is_tensor(t) else t for t in meta_v.meta_args(M, False))
                    meta_v.prepare_tensors(tok)
                    margs = meta_v.meta_args(M, False)

                    def ref_fn(xx=x):   # also the hybrid path's base_fn (x -> y)
                        xq, xs = ops.scaled_fp4_quant(xx, xg_t)
                        return ops.cutlass_scaled_fp4_mm(xq, w_q_sw, xs, w_sf_sw, alpha, torch.bfloat16)

                    def stock_fn():
                        y = ref_fn()
                        lora_shrink(x, a_st, hbuf, *margs, 1.0)
                        lora_expand(hbuf, b_st, y, *margs, offset_start=0, add_inputs=True)
                        return y

                    # parity: quantization and no-adapter GEMM against vLLM, fused against the fp32 reference
                    y0 = op(x, margs0).clone()
                    xq_v, xs_v = ops.scaled_fp4_quant(x, xg_t, is_sf_swizzled_layout=False)
                    q_codes = float((op.xq[:M] != xq_v).float().mean())
                    q_scales = float((op.sxq[:M].view(torch.uint8) != xs_v.view(torch.uint8)).float().mean())
                    xq_vs, xs_vs = ops.scaled_fp4_quant(x, xg_t)
                    y_v = ops.cutlass_scaled_fp4_mm(xq_vs, w_q_sw, xs_vs, w_sf_sw, alpha, torch.bfloat16)
                    gemm_diff = float((y0.view(torch.int16) != y_v.view(torch.int16)).float().mean())
                    gemm_rel = rel_err(y0, y_v)
                    y1 = op(x, margs).clone()
                    ref = F.reference(x, tok.cpu(), w_q, w_sf, w_global, x_global, A_ref, B_ref, slices,
                                      xq=op.xq[:M].clone(), xsf=op.sxq[:M].clone())
                    errs = {"fused": rel_err(y1, ref), "stock": rel_err(stock_fn(), ref)}
                    times = {"ref": timer.graph_time(ref_fn), "stock": timer.graph_time(stock_fn),
                             "base": timer.graph_time(lambda: op(x, margs0)),
                             "fused": timer.graph_time(lambda: op(x, margs, hybrid_m=1 << 30))}
                    if M > 64:   # the plugin's path above 64 tokens: vendor GEMM, then our shrink and expand-add
                        errs["hybrid"] = rel_err(op(x, margs, base_fn=ref_fn), ref)
                        times["hybrid"] = timer.graph_time(lambda: op(x, margs, base_fn=ref_fn))
                    rec = {**info, "model": a.model, "proj": proj, "N": N, "K": K, "r": r, "M": M, "n": n,
                           "t_us": {k: round(v, 2) for k, v in times.items()},
                           "parity": {"q_codes": q_codes, "q_scales": q_scales, "gemm_diff": gemm_diff,
                                      "gemm_rel": float(f"{gemm_rel:.3g}")},
                           "rel_err": {k: float(f"{v:.3g}") for k, v in errs.items()},
                           "ok": {k: v <= 2e-2 for k, v in errs.items() if k != "stock"},
                           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                    print(json.dumps({k: rec[k] for k in ("lora_fmt", "proj", "r", "M", "n", "t_us", "parity", "rel_err")}),
                          flush=True)
                    bad += not all(rec["ok"].values())
                    with out.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
    print("PASS" if not bad else f"FAIL: {bad} cases off their reference by > 2e-2", flush=True)
    return 1 if a.strict and bad else 0


def tune_base(a, F, ops, timer, info, meta_v, out) -> int:
    """GEMM tile search for the no-adapter batch (the base GEMM; the shrink then only quantizes x)."""
    import itertools

    import triton
    cands = [dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=w, num_stages=st, GROUP_M=8)
             for (bm, bn), bk, w, st in itertools.product(
                 [(16, 64), (32, 64), (32, 128), (64, 64), (64, 128), (64, 256), (128, 64), (128, 128), (128, 256)],
                 [128, 256], [4, 8], [2, 3, 4])]
    ms = [int(m) for m in a.ms.split(",")]
    for proj, (K, slices) in SHAPES[a.model].items():
        N, S = sum(slices), len(slices)
        w = (torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16)
        w_global = F.global_scale_for(w.float())
        wg_t = torch.tensor([w_global], dtype=torch.float32, device="cuda")
        w_q_sw, w_sf_sw = ops.scaled_fp4_quant(w, wg_t)
        w_q, w_sf = ops.scaled_fp4_quant(w, wg_t, is_sf_swizzled_layout=False)
        r = 16
        a_st = tuple(torch.zeros(a.pool, 1, r, K, dtype=torch.bfloat16, device="cuda") for _ in range(S))
        b_st = tuple(torch.zeros(a.pool, 1, n_s, r, dtype=torch.bfloat16, device="cuda") for n_s in slices)
        for M in ms:
            x = (torch.randn(M, K, device="cuda") * 2).to(torch.bfloat16)
            x_global = F.global_scale_for(x.float())
            xg_t = torch.tensor([x_global], dtype=torch.float32, device="cuda")
            alpha = torch.tensor([1.0 / (x_global * w_global)], dtype=torch.float32, device="cuda")
            op = F.VllmFusedNVFP4LoRA(w_q, w_sf, w_global, x_global, a_st, b_st, M)
            meta_v.prepare_tensors(torch.full((M,), -1, dtype=torch.int32, device="cuda"))
            margs = meta_v.meta_args(M, False)

            def ref_fn():
                xq, xs = ops.scaled_fp4_quant(x, xg_t)
                return ops.cutlass_scaled_fp4_mm(xq, w_q_sw, xs, w_sf_sw, alpha, torch.bfloat16)

            t_ref = timer.graph_time(ref_fn)
            y_ref = ref_fn()
            cs, cg0 = F.pick_cfg(M, N, K, r, slices)
            res = []
            for cg in [cg0] + cands:
                if any(n_s % cg["BLOCK_N"] for n_s in slices) or cg["BLOCK_M"] > max(16, triton.next_power_of_2(M)) * 2:
                    continue
                try:
                    y = op(x, margs, cfg_s=cs, cfg_g=cg).clone()
                    if rel_err(y, y_ref) > 1e-2:
                        continue
                    res.append((timer.graph_time(lambda: op(x, margs, cfg_s=cs, cfg_g=cg)), cg))
                except Exception:   # noqa: BLE001 -- out of shared memory etc.
                    continue
            res.sort(key=lambda t: t[0])
            best_t, best = res[0]
            rec = {**info, "model": a.model, "proj": proj, "N": N, "K": K, "M": M, "t_ref": round(t_ref, 2),
                   "t_default": round(next(t for t, c in res if c is cg0), 2) if any(c is cg0 for _, c in res) else None,
                   "t_best": round(best_t, 2), "best": best, "top5": [(round(t, 2), c) for t, c in res[:5]],
                   "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            print(json.dumps({k: rec[k] for k in ("proj", "M", "t_ref", "t_default", "t_best", "best")}), flush=True)
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
