"""HOST-OVERHEAD (a diagnostic, not a registered test; 2026-10-25, mock reviews R1 v37 M4 / R3 v38 W1: at vLLM's default capture
the plugin trails on the H100 beyond light load, and its steps above the 512-token capture bound cost it far more than they cost
vLLM: 42.5 -> 68.5 ms TPOT at 16 requests/s against vLLM's 44.8 -> 54.5): are those eager steps host-bound?
Per projection of Qwen3.5-9B (one layer's qkv, qkvz, o, gate_up, down; 32 adapter slots, tokens spread over 32 adapters) and
M in --ms (the eager step sizes), each path of the plugin's op (fused, hybrid = vLLM's GEMM + our LoRA kernels, hybrids = vLLM's
GEMM + vLLM's LoRA kernels, all through VllmFusedFP8LoRA._run with the lean base as base_fn) and vLLM's own sequence (quantize +
CUTLASS + lora_shrink + lora_expand, called directly: a lower bound on vLLM's host cost, which adds its Punica wrapper):
  host_us  host time per call, enqueued behind a GPU sleep so the GPU never throttles the loop
  gpu_us   GPU time per call (CUDA-graph replays)
A path is host-bound in eager steps when host_us > gpu_us.
    python bench/host_overhead.py --out results/host_overhead_<gpu>.jsonl [--ms 512,1024,2048] [--calls 100]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ms", default="512,1024,2048")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--model", default="qwen35_9b")
    a = ap.parse_args()

    import torch
    import pdl_k2_bench as B
    from lowbit_lora.kernels import fp8_lora_vllm as F

    ms = [int(m) for m in a.ms.split(",")]
    bank = B.Bank(max(ms))
    env = B.Env()
    env.set_tuned(False)
    projs = [p for p in ("qkv", "qkvz", "o", "gate_up", "down") if p in B.MODELS[a.model] or p in B.MODELS["qwen3_8b"]]
    model_of = lambda p: a.model if p in B.MODELS[a.model] else "qwen3_8b"   # noqa: E731
    info = {"gpu": torch.cuda.get_device_name(), "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID", ""),
            "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def host_us(fn, n):
        fn()
        torch.cuda.synchronize()
        torch.cuda._sleep(int(3e9))           # a GPU spin of about 2 s: the loop below only enqueues
        t = time.perf_counter()
        for _ in range(n):
            fn()
        dt = time.perf_counter() - t
        torch.cuda.synchronize()
        return 1e6 * dt / n

    def gpu_us(fn):
        return F._graph_us(fn)

    rows = []
    for p in projs:
        mdl = model_of(p)
        op, a_st, b_st = bank.op(mdl, p, 64, 32, "bf16")
        for M in ms:
            _, margs = B.make_meta(32, bank.max_m, M, 32)
            x = torch.randn(M, op.K, device="cuda").to(torch.bfloat16)
            vendor, stock, _dual = env.fns(op, a_st, b_st, x, margs)
            op.base_fn = vendor
            paths = {"vllm": stock, "base_only": vendor,
                     "fused": lambda: op._run(x, margs, hybrid=False),
                     "hybrid": lambda: op._run(x, margs, hybrid=True),
                     "hybrids": lambda: op._run(x, margs, hybrid="s")}
            # the plugin's paths again with LOWBIT_FAST_LAUNCH's direct launches (fp8_lora_vllm._run_fast)
            for name in ("fused", "hybrid"):
                paths[name + "_fast"] = paths[name]
            for name, fn in paths.items():
                F._FAST_LAUNCH = name.endswith("_fast")
                try:
                    h = host_us(fn, a.calls)
                    g = gpu_us(fn)
                    r = {**info, "model": mdl, "proj": p, "K": op.K, "N": op.N, "M": M, "path": name,
                         "host_us": round(h, 1), "gpu_us": round(g, 1), "host_bound": h > g}
                except Exception as e:   # noqa: BLE001
                    torch.cuda.synchronize()
                    r = {**info, "model": mdl, "proj": p, "M": M, "path": name, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                F._FAST_LAUNCH = False
                rows.append(r)
                print(json.dumps(r), flush=True)
            del x, margs
        # --check: the fast launches against Triton's own, bit for bit, at irregular sizes (both paths, decode sizes too)
        for M in (1, 3, 16, 17, 33, 100, 513, 777, 1024, 1500, 2048):
            if M > bank.max_m:
                continue
            _, margs = B.make_meta(32, bank.max_m, M, min(32, M))
            x = torch.randn(M, op.K, device="cuda").to(torch.bfloat16)
            vendor, _s, _d = env.fns(op, a_st, b_st, x, margs)
            op.base_fn = vendor
            for hyb in (False, True):
                F._FAST_LAUNCH = False
                y0 = op._run(x, margs, hybrid=hyb).clone()
                F._FAST_LAUNCH = True
                y1 = op._run(x, margs, hybrid=hyb).clone()
                y2 = op._run(x, margs, hybrid=hyb).clone()      # the second fast call takes the direct launch
                F._FAST_LAUNCH = False
                same = bool(torch.equal(y0, y1) and torch.equal(y0, y2))
                r = {**info, "model": mdl, "proj": p, "M": M, "path": "check_" + ("hybrid" if hyb else "fused"), "identical": same}
                rows.append(r)
                print(json.dumps(r), flush=True)
            del x, margs
    with Path(a.out).open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    # summary: per M, one layer's projections summed, host against GPU
    print("\n== one layer's projections summed (us): host / gpu")
    for M in ms:
        for name in ("vllm", "base_only", "fused", "fused_fast", "hybrid", "hybrid_fast", "hybrids"):
            rs = [r for r in rows if r.get("M") == M and r.get("path") == name and "host_us" in r and r["proj"] in ("qkv", "o", "gate_up", "down")]
            if rs:
                print(f"  M {M:5d} {name:9s} host {sum(r['host_us'] for r in rs):8.1f}  gpu {sum(r['gpu_us'] for r in rs):8.1f}")
    chk = [r for r in rows if r.get("path", "").startswith("check_")]
    print(f"fast-launch check: {sum(r['identical'] for r in chk)} of {len(chk)} identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
