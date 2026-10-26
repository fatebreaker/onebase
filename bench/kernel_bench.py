"""P0-KERNEL (docs/REGISTRY.md): the fused FP8 + LoRA prototype against vLLM's stock four-kernel path (GPU).

    python bench/kernel_bench.py --out results/p0_kernel.jsonl [--models qwen3_8b,llama31_8b] [--ranks 16,64,128]
                                 [--ms 1,16,32,64,128,256,1024,4096] [--quick]
Per (projection, r, M), all with one adapter on every token:
  ref     vLLM scaled_fp8_quant (per token) + cutlass_scaled_mm               (no LoRA: the merged cost)
  stock   ref + vLLM lora_shrink + lora_expand (Punica)                        (four kernels)
  full_d  our shrink_absmax (split-K) + gemm_lora with on-the-fly x quant     (two kernels, decode form)
  full_p  our shrink_absmax writing x_q + gemm_lora on x_q                     (two kernels, prefill form)
  vendor  our shrink_absmax writing x_q + cutlass_scaled_mm + vLLM lora_expand (three kernels)
  nolora  our two kernels with the LoRA epilogue off (our GEMM against CUTLASS; descriptive)
Correctness first: every variant against the fp32 reference (dequantized FP8 GEMM + exact LoRA); a variant's time
counts only if ||y - y_ref|| / ||y_ref|| <= 2e-2. Timing: CUDA graph per op sequence, 20 warm-up replays, then the
median of 200 replays, L2 flushed (256 MB write) before each. rho = 1 - (t_proto - t_ref) / (t_stock - t_ref) with
t_proto the fastest correct prototype variant.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

SHAPES = {   # projection -> (K, slice output sizes) for the dense models (q/k/v and gate/up are fused in vLLM)
    "qwen3_8b": {"qkv": (4096, (4096, 1024, 1024)), "o": (4096, (4096,)), "gate_up": (4096, (12288, 12288)),
                 "down": (12288, (4096,))},
    "llama31_8b": {"qkv": (4096, (4096, 1024, 1024)), "o": (4096, (4096,)), "gate_up": (4096, (14336, 14336)),
                   "down": (14336, (4096,))},
}


def gpu_info() -> dict:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader", "-i", "0"],
                       capture_output=True, text=True).stdout.strip()
    name, _, cc = q.partition(",")
    return {"gpu": name.strip(), "cc": cc.strip()}


class Timer:
    def __init__(self, warm: int = 20, reps: int = 200):
        self.warm, self.reps = warm, reps
        self.flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")

    def graph_time(self, fn) -> float:
        """Median microseconds of one replay of fn captured in a CUDA graph, L2 flushed before each replay."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        for _ in range(self.warm):
            g.replay()
        ts = []
        for _ in range(self.reps):
            self.flush.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            ts.append(1000 * e0.elapsed_time(e1))
        return statistics.median(ts)


def rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.float() - ref).norm() / ref.norm().clamp_min(1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen3_8b,llama31_8b")
    ap.add_argument("--ranks", default="16,64,128")
    ap.add_argument("--ms", default="1,16,32,64,128,256,1024,4096")
    ap.add_argument("--quick", action="store_true", help="smoke test: default configs, 20 replays, no tuning")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels.fp8_lora import FusedFP8LoRA, quantize_rows_fp8, reference
    torch.manual_seed(0)
    timer = Timer(warm=5 if a.quick else 20, reps=20 if a.quick else 200)
    import hashlib
    src = Path(__file__).resolve().parents[1] / "lowbit_lora/kernels/fp8_lora.py"
    info = {**gpu_info(), "torch": torch.__version__, "job": os.environ.get("SLURM_JOB_ID", ""),
            "kernel_sha": hashlib.sha256(src.read_bytes()).hexdigest()[:12],
            "precise_acc": os.environ.get("LOWBIT_PRECISE", "1") != "0"}
    ms = [int(m) for m in a.ms.split(",")]
    max_m = max(ms)
    meta = LoRAKernelMeta.make(1, max_m, device="cuda")
    done = set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for model in a.models.split(","):
        for proj, (K, slices) in SHAPES[model].items():
            N = sum(slices)
            key = (K, slices)
            w = torch.randn(N, K, device="cuda") * 0.02
            s_w = (w.abs().max() / 448.0).reshape(1).float()
            w_q = (w / s_w).to(torch.float8_e4m3fn)
            w_t = w_q.t()                                            # [K, N] column-major, as vLLM stores it
            for r in [int(x) for x in a.ranks.split(",")]:
                if (key, r) in done:                                  # same shape in another model: reuse
                    continue
                done.add((key, r))
                S = len(slices)
                a_cat = (torch.randn(S * r, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
                b = (torch.randn(N, r, device="cuda") * 0.02).to(torch.bfloat16)
                scale = 2.0
                a_st = tuple(a_cat[s * r:(s + 1) * r].reshape(1, 1, r, K).contiguous() for s in range(S))
                offs = [sum(slices[:s]) for s in range(S)]
                b_st = tuple(b[offs[s]:offs[s] + slices[s]].reshape(1, 1, slices[s], r).contiguous() for s in range(S))
                fused = FusedFP8LoRA(w_q, s_w, a_cat, b, slices, scale, max_m)
                plain = FusedFP8LoRA(w_q, s_w, a_cat, b, slices, scale, max_m, has_lora=False)
                a8, sa = quantize_rows_fp8(a_cat)                    # descriptive: FP8-stored adapter (A and B)
                b8, sb = quantize_rows_fp8(b)
                fused8 = FusedFP8LoRA(w_q, s_w, a8, b8, slices, scale, max_m, a_scale=sa, b_scale=sb)
                buf = torch.empty(S * max_m * r, dtype=torch.float32, device="cuda")
                for M in ms:
                    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                    meta.prepare_tensors(torch.zeros(M, dtype=torch.int32, device="cuda"))
                    margs = meta.meta_args(M, False)
                    hbuf = buf[: S * M * r].view(S, M, r)

                    def ref_fn():
                        xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                        return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)

                    def stock_fn():
                        xq, sx = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                        y = ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
                        lora_shrink(x, a_st, hbuf, *margs, scale)
                        lora_expand(hbuf, b_st, y, *margs, offset_start=0, add_inputs=True)
                        return y

                    def vendor_fn():
                        fused.shrink(x, "prefill")                   # writes x_q, s_x and h (SPLIT_K = 1)
                        y = ops.cutlass_scaled_mm(fused.xq[:M], w_t, fused.sx[:M].view(M, 1), s_w, torch.bfloat16)
                        lora_expand(fused.shrink_buffer(M), b_st, y, *margs, offset_start=0, add_inputs=True)
                        return y

                    y_ref = reference(x, w_q, s_w, a_cat, b, slices, scale)
                    y_ref_nolora = reference(x, w_q, s_w, a_cat, torch.zeros_like(b), slices, scale)
                    rec = {**info, "model": model, "proj": proj, "N": N, "K": K, "slices": list(slices), "r": r, "M": M}
                    modes = ["decode", "prefill"] if M <= 256 else ["prefill"]
                    if not a.quick:
                        for mode in modes:
                            fused.tune(x, mode)
                            plain.tune(x, mode)
                            fused8.tune(x, mode)
                    errs = {"stock": rel_err(stock_fn(), y_ref), "ref_nolora": rel_err(ref_fn(), y_ref_nolora)}
                    times = {"ref": timer.graph_time(ref_fn), "stock": timer.graph_time(stock_fn)}
                    for mode in modes:
                        tag = "full_d" if mode == "decode" else "full_p"
                        errs[tag] = rel_err(fused(x, mode).clone(), y_ref)
                        times[tag] = timer.graph_time(lambda: fused(x, mode))
                        errs["nolora_" + mode[0]] = rel_err(plain(x, mode).clone(), y_ref_nolora)
                        times["nolora_" + mode[0]] = timer.graph_time(lambda: plain(x, mode))
                    y_ref8 = reference(x, w_q, s_w, a8, b8, slices, scale, a_scale=sa, b_scale=sb)
                    for mode in modes:                               # descriptive, outside rho (other numerics)
                        tag = "fp8ad_" + mode[0]
                        errs[tag] = rel_err(fused8(x, mode).clone(), y_ref8)
                        times[tag] = timer.graph_time(lambda: fused8(x, mode))
                    errs["vendor"] = rel_err(vendor_fn().clone(), y_ref)
                    times["vendor"] = timer.graph_time(vendor_fn)
                    ok = {k: v <= 2e-2 for k, v in errs.items()}
                    protos = [k for k in ("full_d", "full_p", "vendor") if k in times and ok.get(k)]
                    best = min(protos, key=lambda k: times[k]) if protos else None
                    den = times["stock"] - times["ref"]
                    rec.update({"t_us": {k: round(v, 2) for k, v in times.items()},
                                "rel_err": {k: float(f"{v:.3g}") for k, v in errs.items()}, "ok": ok, "best": best,
                                "rho": round(1 - (times[best] - times["ref"]) / den, 3) if best and den > 0 else None,
                                "stock_overhead": round(den / times["ref"], 3),
                                "cfg": {m: [fused.cfg_shrink.get((M, m)), fused.cfg_gemm.get((M, m))] for m in modes},
                                "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    print(json.dumps({k: rec[k] for k in ("model", "proj", "r", "M", "t_us", "best", "rho",
                                                          "stock_overhead")}), flush=True)
                    with out.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
