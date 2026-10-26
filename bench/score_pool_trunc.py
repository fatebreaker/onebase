"""SERVE-POOL-TRUNC (docs/REGISTRY.md, registered before any data; this scorer written before any data): SERVE-POOL-LONG's
128-adapter pool with rank-16 adapters (an SVD truncation's size) in 128 bf16 slots -- the slot memory of 32 rank-64 ones, every
adapter resident -- against the references of the same jobs (jobs/serve_pool_trunc.sbatch -> results/serve_pool_trunc.jsonl;
each configuration's newest row per seed and rate, so a requeued job's rerun replaces the earlier one, and the rate list's
second "inf" pass, the measured one, replaces its warm-up).
  prediction  at 4 requests/s, the stock path with 128 rank-16 slots has a lower p99 time to first token than the stock path with
              32 rank-64 slots, in every seed (truncation buys slots, as storage does)
Descriptive: p99 TTFT, median TPOT and output throughput per configuration and rate (mean and range over the seeds); rank 16 in
128 slots (stock and ours) against FP8-stored rank-64 adapters in 64 slots.
    python3 bench/score_pool_trunc.py [--json results/pool_trunc_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
CFGS = (("stock", 128, 16), ("fused", 128, 16), ("stock", 32, 64), ("fused-fp8ad", 64, 64))
RATES = ("2", "4", "6", "inf")
SEEDS = (0, 1, 2)


def load(path: Path) -> dict:
    out: dict = {}
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    for r in sorted(rows, key=lambda r: r.get("time", "")):
        out[(r["mode"], int(r.get("max_loras") or 0), int(r.get("rank") or 0), int(r["seed"]), str(r["rate"]))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = load(LB / "results/serve_pool_trunc.jsonl")
    seeds = [s for s in SEEDS if all((m, k, r, s, q) in d for m, k, r in CFGS for q in RATES)]
    print(f"complete seeds: {seeds}")
    desc = {}
    for m, k, r in CFGS:
        for q in RATES:
            xs = [d[(m, k, r, s, q)] for s in seeds]
            if xs:
                desc[f"{m}:{k}:r{r}@{q}"] = {f: {"mean": sum(x[f] for x in xs) / len(xs), "min": min(x[f] for x in xs),
                                                "max": max(x[f] for x in xs)}
                                            for f in ("p99_ttft_ms", "median_tpot_ms", "output_throughput")}
    per_seed = {s: {"stock_128_r16": d[("stock", 128, 16, s, "4")]["p99_ttft_ms"], "stock_32_r64": d[("stock", 32, 64, s, "4")]["p99_ttft_ms"],
                    "fp8_64_r64": d[("fused-fp8ad", 64, 64, s, "4")]["p99_ttft_ms"], "fused_128_r16": d[("fused", 128, 16, s, "4")]["p99_ttft_ms"]}
                for s in seeds}
    v = {"seeds": seeds, "per_seed": per_seed,
         "met": (len(seeds) == len(SEEDS) and all(x["stock_128_r16"] < x["stock_32_r64"] for x in per_seed.values())) if seeds else None}
    for s, x in per_seed.items():
        print(f"seed {s}: p99 TTFT at 4 req/s (s): " + ", ".join(f"{k} {val / 1000:.2f}" for k, val in x.items()))
    print("verdict:", json.dumps({kk: vv for kk, vv in v.items() if kk != "per_seed"}))
    if a.json:
        Path(a.json).write_text(json.dumps({"verdict": v, "descriptive": desc}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
