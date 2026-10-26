"""GEMM-BASELINES-L40S (docs/REGISTRY.md, registered before any data; mock reviews v16 R1 question 2 and v17 R3 question 3):
is the plugin's faster base GEMM at small batches on the L40S a property of our kernel, or of vLLM's CUTLASS choice on sm_89?
At Qwen3-8B's projection shapes (Qwen3.5-9B's dense ones; kernel_bench.py's SHAPES) and M in --ms, without adapters, each from
bf16 x to bf16 y with per-token activation and per-tensor weight scales (vLLM's online FP8):
  cutlass     vLLM scaled_fp8_quant (per token) + cutlass_scaled_mm                 (vLLM's FP8 linear on sm_89; kernel_bench ref)
  ours_d      the plugin's kernels with the LoRA epilogue off, decode form           (kernel_bench.py's nolora_d)
  ours_p      the same, prefill form                                                 (nolora_p)
  cublas_row  vLLM scaled_fp8_quant (per token) + torch._scaled_mm, row-wise scales  (cuBLASLt; recorded as an error if unsupported)
  cublas_fb   vLLM scaled_fp8_quant (per token) + torch._scaled_mm with unit scales, fp32 out, then both scales applied
              (vLLM's torch fallback for per-token scales, w8a8_utils)
  cublas_t    torch._scaled_mm alone on pre-quantized x, unit scales (cuBLASLt's GEMM alone: a lower bound, not a full op)
  bf16        x @ W^T in bf16 (cuBLAS)
  plugin      the plugin's own class (fp8_lora_vllm.VllmFusedFP8LoRA, its fixed pick_cfg configs, no dispatch table) on
              tokens mapped to no adapter: the base GEMM the plugin serves (kernel_breakdown.py's ours_base); amendment
              -PLUGIN (post hoc). plugin_r16: the same op with one rank-16 adapter on every token (descriptive)
Timing as kernel_bench.py (CUDA graph per op, L2 flushed before each replay, median of 200 replays); a variant's time counts only
if its relative error against the fp32 reference is <= 2e-2 (cublas_t's after the scales are applied outside the timed op).
    python bench/gemm_baselines.py --out results/gemm_baselines_l40s.jsonl [--ms 1,4,16,64,256] [--no-tuned] [--plugin]
--no-tuned drops ours_d / ours_p (eager-tuned per kernel_bench.py: on a shared pod's CPU the eager timing picked configs the
plugin never runs; amendment -PLUGIN)."""
from __future__ import annotations

import argparse
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
    ap.add_argument("--ms", default="1,4,16,64,256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--plugin", action="store_true")
    a = ap.parse_args()
    from vllm import _custom_ops as ops

    from vllm.lora.ops.triton_ops import LoRAKernelMeta

    from lowbit_lora.kernels import fp8_lora_vllm as F
    from lowbit_lora.kernels.fp8_lora import FusedFP8LoRA, reference
    meta = LoRAKernelMeta.make(1, max(int(m) for m in a.ms.split(",")), device="cuda")
    torch.manual_seed(0)
    timer = Timer()
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", "")}
    ms = [int(m) for m in a.ms.split(",")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    one = torch.ones(1, device="cuda", dtype=torch.float32)
    for proj, (K, slices) in SHAPES[a.model].items():
        N = sum(slices)
        w = torch.randn(N, K, device="cuda") * 0.02
        s_w = (w.abs().max() / 448.0).reshape(1).float()
        w_q = (w / s_w).to(torch.float8_e4m3fn)
        w_t = w_q.t()                                                # [K, N] column-major, as vLLM stores it
        w_bf = (w_q.float() * s_w).to(torch.bfloat16)
        s_wn = s_w.reshape(1, 1).expand(1, N).contiguous()
        r, S = 16, len(slices)
        a_cat = (torch.randn(S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
        b = torch.zeros(N, r, device="cuda", dtype=torch.bfloat16)
        plain = FusedFP8LoRA(w_q, s_w, a_cat, b, slices, 1.0, max(ms), has_lora=False)
        if a.plugin:                                                 # vLLM's stacked buffers, as kernel_breakdown.py
            b_r = (torch.randn(N, r, device="cuda") * 0.02).to(torch.bfloat16)
            a_st = tuple(a_cat[s * r:(s + 1) * r].reshape(1, 1, r, K).contiguous() for s in range(S))
            offs = [sum(slices[:s]) for s in range(S)]
            b_st = tuple(b_r[offs[s]:offs[s] + slices[s]].reshape(1, 1, slices[s], r).contiguous() for s in range(S))
            op = F.VllmFusedFP8LoRA(w_q, s_w, a_st, b_st, max(ms))
        for M in ms:
            x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
            y_ref = reference(x, w_q, s_w, a_cat, b, slices, 1.0)
            xq0, sx0 = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

            def cutlass():
                xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)

            def cublas_row():
                xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                return torch._scaled_mm(xq, w_t, scale_a=sx, scale_b=s_wn, out_dtype=torch.bfloat16)

            def cublas_fb():
                xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                y = torch._scaled_mm(xq, w_t, scale_a=one, scale_b=one, out_dtype=torch.float32)
                return (y * sx * s_w).to(torch.bfloat16)

            def cublas_t():
                return torch._scaled_mm(xq0, w_t, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

            def bf16():
                return x @ w_bf.t()

            rec = {**info, "model": a.model, "proj": proj, "N": N, "K": K, "M": M}
            times, errs, fails = {}, {}, {}
            modes = [] if a.no_tuned else ["decode", "prefill"] if M <= 256 else ["prefill"]
            for mode in modes:
                plain.tune(x, mode)
            variants = {"cutlass": cutlass, "cublas_row": cublas_row, "cublas_fb": cublas_fb, "bf16": bf16}
            for mode in modes:
                variants["ours_" + mode[0]] = (lambda m=mode: plain(x, m))
            if a.plugin:
                margs0 = op._dummy_meta(M, -1)                       # no adapter token: the plugin's base GEMM alone
                variants["plugin"] = lambda: op._run(x, margs0)
            y_r16 = None
            if a.plugin:
                meta.prepare_tensors(torch.zeros(M, dtype=torch.int32, device="cuda"))
                margs1 = meta.meta_args(M, False)
                y_r16 = reference(x, w_q, s_w, a_cat, b_r, slices, 1.0)
            for k, fn in variants.items():
                try:
                    errs[k] = rel_err(fn().clone(), y_ref)
                    times[k] = timer.graph_time(fn)
                except Exception as e:                               # noqa: BLE001 (an unsupported path is a result)
                    fails[k] = f"{type(e).__name__}: {str(e)[:200]}"
            if a.plugin:
                try:
                    errs["plugin_r16"] = rel_err(op._run(x, margs1).clone(), y_r16)
                    times["plugin_r16"] = timer.graph_time(lambda: op._run(x, margs1))
                except Exception as e:                               # noqa: BLE001
                    fails["plugin_r16"] = f"{type(e).__name__}: {str(e)[:200]}"
            try:
                errs["cublas_t"] = rel_err((cublas_t().float() * sx0 * s_w).to(torch.bfloat16), y_ref)
                times["cublas_t"] = timer.graph_time(cublas_t)
            except Exception as e:                                   # noqa: BLE001
                fails["cublas_t"] = f"{type(e).__name__}: {str(e)[:200]}"
            ok = {k: v <= 2e-2 for k, v in errs.items()}
            rec.update({"t_us": {k: round(v, 2) for k, v in times.items()},
                        "rel_err": {k: float(f"{v:.3g}") for k, v in errs.items()}, "ok": ok, "fail": fails,
                        "cfg": {m: plain.cfg_gemm.get((M, m)) for m in modes},
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            print(json.dumps({k: rec[k] for k in ("proj", "M", "t_us", "ok", "fail")}), flush=True)
            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
