"""SERVE-POOL-LONG (docs/REGISTRY.md, registered 2026-10-16 15:42 UTC): the 128-adapter pool with 2,000-request traces and
request-sequence seeds 0, 1, 2 (jobs/serve_pool3.sbatch -> results/serve_pool_long.jsonl; each configuration's newest row
per seed and rate, so a trace rerun after an interruption replaces the earlier one, and the rate list's second "inf" pass, the
measured one, replaces its warm-up).
  prediction  at 4 requests/s, NVFP4-stored adapters in 112 slots (fused-fp4ad, 112) have a lower p99 time to first token than
              the stock path with 32 slots, in every seed
Descriptive: p99 TTFT, median TPOT and output throughput per configuration and rate, mean and range over the seeds; the stock
path with 64 slots against ours.
    python3 bench/score_pool_long.py [--json results/pool_long_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
CFGS = (("nolora", 0), ("stock", 32), ("stock", 64), ("fused", 32), ("fused-fp8ad", 64), ("fused-fp4ad", 112), ("fused-fp4ad", 32))
OPTIONAL = {("stock", 64)}   # descriptive; on the L40S its server does not start (0.44 GiB of KV cache left after 64 bf16 slots)
RATES = ("2", "4", "6", "inf")
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    out: dict = {}
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    for r in sorted(rows, key=lambda r: r.get("time", "")):
        out[(r["mode"], int(r.get("max_loras") or 0), int(r["seed"]), str(r["rate"]))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = load(LB / "results/serve_pool_long.jsonl")
    seeds = [s for s in SEEDS if all((m, k, s, r) in d for m, k in CFGS if (m, k) not in OPTIONAL for r in RATES)]
    print(f"complete seeds: {seeds}")
    desc = {}
    for m, k in CFGS:
        for r in RATES:
            xs = [d[(m, k, s, r)] for s in seeds if (m, k, s, r) in d]
            if xs:
                desc[f"{m}:{k}@{r}"] = {f: {"mean": sum(x[f] for x in xs) / len(xs), "min": min(x[f] for x in xs),
                                           "max": max(x[f] for x in xs)}
                                       for f in ("p99_ttft_ms", "median_tpot_ms", "output_throughput")}
    per_seed = {s: {"fp4_112": d[("fused-fp4ad", 112, s, "4")]["p99_ttft_ms"], "stock_32": d[("stock", 32, s, "4")]["p99_ttft_ms"]}
                for s in seeds}
    v = {"seeds": seeds, "per_seed": per_seed,
         "met": (len(seeds) == len(SEEDS) and all(x["fp4_112"] < x["stock_32"] for x in per_seed.values())) if seeds else None}
    for s, x in per_seed.items():
        print(f"seed {s}: p99 TTFT at 4 req/s: NVFP4-stored/112 {x['fp4_112'] / 1000:.2f} s, stock/32 {x['stock_32'] / 1000:.2f} s")
    print("verdict:", json.dumps({kk: vv for kk, vv in v.items() if kk != "per_seed"}))
    if a.json:
        Path(a.json).write_text(json.dumps({"verdict": v, "descriptive": desc}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
