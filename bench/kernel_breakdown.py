"""Per-kernel breakdown of the LoRA paths at decode (MLSys v2 mock review: R1 asked where the Triton route stops on
datacenter GPUs, R3 for a bytes model that predicts where fusion pays). Per projection, rank, batch size and number of
distinct adapters, CUDA-graph timings (bench/kernel_bench.py's Timer: median replay, L2 flushed) of every kernel alone
and of every path, with the bytes each kernel moves:
  quant      vLLM's per-token FP8 quantize                    cutlass   CUTLASS FP8 GEMM on the quantized x
  ref        quant + cutlass (no LoRA: the merged cost)
  v_shrink   vLLM's lora_shrink alone                         v_expand  vLLM's lora_expand (add into y) alone
  stock      ref + v_shrink + v_expand
  k1h        our shrink alone, hybrid form (adapter segments) k2h       our expand-add alone (the GEMM kernel, BASE=False)
  hybrid     k1h + ref + k2h (what the H100 dispatch keeps)
  k1f        K1 alone (segments + the all-token quantize)     k2f       K2 alone (FP8 GEMM with the LoRA epilogue)
  fused      k1f + k2f                                        ours_base our K1 + K2 with no adapter token (our base GEMM)
and, once per GPU, the floor of a CUDA graph with one and with two dependent empty kernels, and the achievable DRAM
bandwidth (a 1 GiB device copy). Kernels are timed alone by replacing the others' launches with no-ops inside the
captured graph; their inputs hold the values of a full run.
    python bench/kernel_breakdown.py [--model qwen3_8b] [--ranks 16,64] [--ms 1,16,64] [--ns 1,8] --out results/kernel_breakdown.jsonl
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
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kernel_bench import SHAPES, Timer, gpu_info  # noqa: E402


@triton.jit
def _noop(X):
    tl.store(X, tl.load(X) + 1)


class _NoLaunch:
    """Stands in for a @triton.jit kernel: kernel[grid](...) launches nothing."""

    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3_8b")
    ap.add_argument("--ranks", default="16,64")
    ap.add_argument("--ms", default="1,16,64")
    ap.add_argument("--ns", default="1,8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import _custom_ops as ops
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink

    from lowbit_lora.kernels import fp8_lora_vllm as F
    torch.manual_seed(0)
    timer = Timer(warm=10, reps=100)
    info = {**gpu_info(), "job": os.environ.get("SLURM_JOB_ID", ""), "kernel_sha": os.environ.get("LOWBIT_SHA", "")}
    # floors: one and two dependent empty kernels in a graph; achievable DRAM bandwidth
    buf = torch.zeros(1, dtype=torch.float32, device="cuda")
    t_e1 = timer.graph_time(lambda: _noop[(1,)](buf))
    t_e2 = timer.graph_time(lambda: (_noop[(1,)](buf), _noop[(1,)](buf)))
    src = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    t_cp = timer.graph_time(lambda: dst.copy_(src))
    bw = 2 * src.numel() / (t_cp * 1e-6)                        # bytes read + written per second
    del src, dst
    floor = {**info, "kind": "floor", "t_empty1": round(t_e1, 2), "t_empty2": round(t_e2, 2), "bw_GBps": round(bw / 1e9, 1),
             "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(floor), flush=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(floor) + "\n")
    ms = [int(m) for m in a.ms.split(",")]
    ns = [int(n) for n in a.ns.split(",")]
    L = max(8, max(ns))
    meta = LoRAKernelMeta.make(L, max(ms), device="cuda")
    real_shrink, real_gemm = F._shrink_vllm, F._gemm_vllm
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
            hbuf = torch.empty(S * max(ms) * r, dtype=torch.float32, device="cuda")
            for M, n in itertools.product(ms, ns):
                if n > M:
                    continue
                x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
                xq0, sx0 = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
                ybuf = ops.cutlass_scaled_mm(xq0, w_t, sx0, s_w, torch.bfloat16)

                def quant():
                    return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)

                def cutlass():
                    return ops.cutlass_scaled_mm(xq0, w_t, sx0, s_w, torch.bfloat16)

                def vendor(xx):
                    xq, sx = ops.scaled_fp8_quant(xx, use_per_token_if_dynamic=True)
                    return ops.cutlass_scaled_mm(xq, w_t, sx, s_w, torch.bfloat16)
                tok = (torch.arange(M, device="cuda") % n).to(torch.int32)   # n distinct adapters, interleaved (decode)
                meta.prepare_tensors(tok)
                margs = meta.meta_args(M, False)
                margs0 = op._dummy_meta(M, -1)                                # no adapter token: our base GEMM alone
                hb = hbuf[: S * M * r].view(S, M, r)

                def v_shrink():
                    hb.zero_()
                    lora_shrink(x, a_st, hb, *margs, 1.0)

                def v_expand():
                    lora_expand(hb, b_st, ybuf, *margs, offset_start=0, add_inputs=True)

                def stock():
                    y = vendor(x)
                    hb.zero_()
                    lora_shrink(x, a_st, hb, *margs, 1.0)
                    lora_expand(hb, b_st, y, *margs, offset_start=0, add_inputs=True)
                    return y
                y_ref = stock().float()
                rec = {**info, "kind": "proj", "model": a.model, "proj": proj, "K": K, "N": N, "r": r, "M": M, "n": n,
                       "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                t = {}
                t["quant"] = timer.graph_time(quant)
                t["cutlass"] = timer.graph_time(cutlass)
                t["ref"] = timer.graph_time(lambda: vendor(x))
                t["v_shrink"] = timer.graph_time(v_shrink)
                v_shrink()
                t["v_expand"] = timer.graph_time(v_expand)
                t["stock"] = timer.graph_time(stock)
                op.base_fn = vendor
                op.use_tma = False
                t["hybrid"] = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                t["fused"] = timer.graph_time(lambda: op._run(x, margs))
                err = float((op._run(x, margs).float() - y_ref).norm() / y_ref.norm())
                t["ours_base"] = timer.graph_time(lambda: op._run(x, margs0))
                # kernels alone: the other launches replaced by no-ops inside the captured graph
                try:
                    op._run(x, margs)                                    # K1's outputs (codes, h) for K2 alone
                    F._gemm_vllm = _NoLaunch()
                    t["k1f"] = timer.graph_time(lambda: op._run(x, margs))
                    F._gemm_vllm = real_gemm
                    F._shrink_vllm = _NoLaunch()
                    t["k2f"] = timer.graph_time(lambda: op._run(x, margs))
                    F._shrink_vllm = real_shrink
                    op._run(x, margs, hybrid=True)
                    op.base_fn = lambda xx: ybuf                         # no vendor kernels
                    F._gemm_vllm = _NoLaunch()
                    t["k1h"] = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                    F._gemm_vllm = real_gemm
                    F._shrink_vllm = _NoLaunch()
                    t["k2h"] = timer.graph_time(lambda: op._run(x, margs, hybrid=True))
                finally:
                    F._shrink_vllm, F._gemm_vllm = real_shrink, real_gemm
                    op.base_fn = vendor
                n_act = n                                                 # adapters whose weights are read
                by = {"quant": 2 * M * K + M * K + 4 * M,
                      "cutlass": M * K + N * K + 2 * M * N,
                      "v_shrink": 2 * M * K + n_act * S * r * K * 2 + 4 * S * M * r,
                      "v_expand": 4 * S * M * r + n_act * N * r * 2 + 4 * M * N,
                      "k1h": 2 * M * K + n_act * S * r * K * 2 + 4 * S * M * r,
                      "k2h": 4 * S * M * r + n_act * N * r * 2 + 4 * M * N,
                      "k1f": 2 * M * K + n_act * S * r * K * 2 + 4 * S * M * r + M * K + 4 * M,
                      "k2f": M * K + N * K + 4 * S * M * r + n_act * N * r * 2 + 2 * M * N,
                      "ours_base": 2 * M * K + M * K + N * K + 2 * M * N}
                rec["t_us"] = {k: round(v, 2) for k, v in t.items()}
                rec["bytes"] = by
                rec["err_fused"] = err
                print(json.dumps({"proj": proj, "r": r, "M": M, "n": n, **rec["t_us"]}), flush=True)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
