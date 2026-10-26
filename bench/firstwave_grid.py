"""FIRSTWAVE-GRID (docs/REGISTRY.md; mock review R3 v38, W3 / change 2: a discriminating test of the first-wave window): the decode
path's exposed shrink as K2's grid grows past the SM count, with everything else fixed. Per projection of Qwen3-8B (qkv, o, gate_up,
down), M in --ms, one rank-64 adapter, K2's tiles forced over BLOCK_N in 32-256 and SPLIT_K in 1-8 (the rest of each configuration from
the decode path's resolved defaults and swept table), each configuration timed in CUDA graphs:
  t_main    K2 without the adapter (lora=False)          t_shrink  the shrink kernel alone (gemm=False)
  t_pdl     the decode path                              t_hgiven  the same K2 with h already computed (shrink=False)
  exposed = t_pdl - t_hgiven; grid = K2's thread blocks (hopper_pdl._geometry: tiles x split-K); n_sm from the device
  first-wave prediction max(0, t_shrink - t_main * min(1, n_sm / grid)); whole-loop prediction max(0, t_shrink - t_main)
Each configuration's output is checked against vLLM's quantize + CUTLASS + LoRA kernels (relative error <= 2e-2).
    python bench/firstwave_grid.py --out results/firstwave_grid_<gpu>.jsonl [--ms 1,16]"""
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
    ap.add_argument("--ms", default="1,16")
    ap.add_argument("--projs", default="qkv,o,gate_up,down")
    ap.add_argument("--rank", type=int, default=64, help="the adapter's rank (the base tiles stay the rank-64 sweep's)")
    ap.add_argument("--base-from", default="", help="-SWEPT: each (projection, M)'s swept best (the headline stage's newest job "
                    "in this decode-path benchmark file) as the base configuration, instead of the decode path's defaults")
    a = ap.parse_args()

    import torch
    import pdl_k2_bench as B
    from kernel_bench import Timer
    from lowbit_lora import hopper_pdl as H

    timer = Timer(warm=5, reps=40)
    bank = B.Bank(16)
    env = B.Env()
    env.set_tuned(False)
    H.load_table(str(Path(os.environ.get("LOWBIT_PDL_CFG", B.REPO / "results/pdl_k2.jsonl"))))
    n_sm = torch.cuda.get_device_properties(0).multi_processor_count
    info = {"gpu": torch.cuda.get_device_name(), "n_sm": n_sm, "job": os.environ.get("SLURM_JOB_ID", ""),
            "torch": torch.__version__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rows = []
    for p in a.projs.split(","):
        op, a_st, b_st = bank.op("qwen3_8b", p, a.rank, 8, "bf16")
        for M in (int(m) for m in a.ms.split(",")):
            _, margs = B.make_meta(8, bank.max_m, M, 1)
            x = torch.randn(M, op.K, device="cuda").to(torch.bfloat16)
            vendor, stock, _ = env.fns(op, a_st, b_st, x, margs)
            ref = stock().float().clone()
            base = H.resolve_cfg(op, M, {})
            if a.base_from:   # FIRSTWAVE-GRID amendment -SWEPT (post hoc): the swept tiles, with BLOCK_N and SPLIT_K forced below
                src = [json.loads(s) for s in Path(a.base_from).read_text().splitlines() if s.strip()]
                src = [s for s in src if s.get("stage") == "headline" and s.get("model") == "qwen3_8b" and s.get("proj") == p
                       and s.get("M") == M and s.get("n") == 1 and s.get("r") == 64 and s.get("pdl_best")]
                if src:
                    job = max(s.get("job", "") for s in src)
                    base = H.resolve_cfg(op, M, next(s for s in src if s.get("job", "") == job)["pdl_best"])
            for bn in (32, 64, 128, 256):
                if any(w % bn for w in op.widths):
                    continue
                for sk in (1, 2, 4, 8):
                    if op.K // sk < int(base["BLOCK_K"]):
                        continue
                    c = dict(base, BLOCK_N=bn, SPLIT_K=sk, num_warps=8 if int(base["BLOCK_M"]) * bn >= 4096 else 4)
                    r = {**info, "model": "qwen3_8b", "proj": p, "K": op.K, "N": op.N, "M": M, "r": a.rank, "cfg": c,
                         "base": "swept" if a.base_from else "defaults"}
                    try:
                        y = H.run_pdl(op, x, margs, cfg=c).float()
                        r["rel_err"] = float(((y - ref).norm() / ref.norm()).item())
                        g = H._geometry(op, x, H.resolve_cfg(op, M, c), True)
                        r["grid"] = int(g["n_tiles"] * g["sk"])
                        r["t_pdl"] = round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c)), 2)
                        H.run_pdl(op, x, margs, cfg=c)                     # h for the no-shrink K2
                        r["t_hgiven"] = round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, shrink=False)), 2)
                        r["t_main"] = round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, lora=False)), 2)
                        r["t_shrink"] = round(timer.graph_time(lambda: H.run_pdl(op, x, margs, cfg=c, gemm=False)), 2)
                        r["exposed"] = round(r["t_pdl"] - r["t_hgiven"], 2)
                        win = r["t_main"] * min(1.0, n_sm / r["grid"])
                        r["window_firstwave"] = round(win, 2)
                        r["pred_firstwave"] = round(max(0.0, r["t_shrink"] - win), 2)
                        r["pred_wholeloop"] = round(max(0.0, r["t_shrink"] - r["t_main"]), 2)
                        r["ok"] = r["rel_err"] <= 2e-2
                    except Exception as e:   # noqa: BLE001
                        torch.cuda.synchronize()
                        r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    rows.append(r)
                    print(json.dumps({k: v for k, v in r.items() if k != "cfg"}), flush=True)
                    with Path(a.out).open("a") as fh:
                        fh.write(json.dumps(r) + "\n")
            del x, margs
        bank.drop_ops()
    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)} of {len(rows)} configurations measured and correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
