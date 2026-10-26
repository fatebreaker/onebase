"""SERVE-LEAN-H100 (docs/REGISTRY.md, registered before any data; written before any data): online serving on the H100 at vLLM's
default CUDA-graph sizes, plugin-pdl-lean (the plugin's decode path with LOWBIT_LEAN_BASE=1) against stock-tuned, SERVE-REPS's
protocol (mlsys jobs/pod_runs/serve_lean_reps_pod.sh -> results/serve_leanreps_h100_<variant>.jsonl), 32 rank-64 tenants, means
over the three starts (request seeds 1-3) of the newest row per start:
  (i)  plugin-pdl-lean's median time per output token below stock-tuned's at every rate (2, 4, 8, 16)
  (ii) its p99 time to first token at most 1.1x stock-tuned's at every rate
Descriptive: per-start ratios; both against SERVE-REPS's rows (graphs to 2,048: results/serve_reps_h100_*.jsonl).
SERVE-LEAN-L40S (registered before any data; --gpu l40s): the same on one L40S with plugin-lean (the plugin without the decode path,
plus LOWBIT_LEAN_BASE=1) at rates 1, 2, 4, 6, against SERVE-REPS's L40S rows (results/serve_reps2_l40s_*.jsonl).
    python3 bench/score_serve_lean.py [--gpu h100|l40s] [--json results/serve_lean_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results"
RATES = ("2", "4", "8", "16")
GPUS = {"h100": ("plugin-pdl-lean", ("2", "4", "8", "16"), "serve_reps", "plugin-pdl"),
        "l40s": ("plugin-lean", ("1", "2", "4", "6"), "serve_reps2", "plugin")}
SEEDS = (1, 2, 3)
R, N = 64, 32


def rows(path: Path) -> dict:
    """{(rate, seed): row}, the newest row per key, for the 64:32 setting; the first rate of each start (8, the warm-up) is the
    first row of that start's rate-8 pair, so only the later of two rate-8 rows of one start counts."""
    out: dict = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("rank") != R or r.get("n_adapters") != N:
            continue
        out[(str(r["rate"]), r.get("seed", 0))] = r          # later rows replace earlier ones (the warm-up 8 is replaced)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--gpu", default="h100", choices=tuple(GPUS))
    a = ap.parse_args()
    global RATES
    lean, RATES, pre, plug = GPUS[a.gpu]
    s = rows(RES / f"serve_leanreps_{a.gpu}_stock-tuned.jsonl")
    p = rows(RES / f"serve_leanreps_{a.gpu}_{lean}.jsonl")
    g = {v: rows(RES / f"{pre}_{a.gpu}_{v}.jsonl") for v in ("stock-tuned", "plugin-pdl")}
    g["plugin-pdl"] = rows(RES / f"{pre}_{a.gpu}_{plug}.jsonl")
    cells, done = {}, True
    for q in RATES:
        if not all((q, sd) in s and (q, sd) in p for sd in SEEDS):
            done = False
            continue
        m = lambda d, k: st.fmean(d[(q, sd)][k] for sd in SEEDS)   # noqa: E731
        c = {"tpot_stock": m(s, "median_tpot_ms"), "tpot_lean": m(p, "median_tpot_ms"),
             "ttft99_stock": m(s, "p99_ttft_ms"), "ttft99_lean": m(p, "p99_ttft_ms")}
        c["tpot_ratio"] = c["tpot_lean"] / c["tpot_stock"]
        c["ttft_ratio"] = c["ttft99_lean"] / c["ttft99_stock"]
        c["tpot_ratio_by_seed"] = [p[(q, sd)]["median_tpot_ms"] / s[(q, sd)]["median_tpot_ms"] for sd in SEEDS]
        if all((q, sd) in g["stock-tuned"] for sd in SEEDS):
            c["tpot_stock_graphs2048"] = st.fmean(g["stock-tuned"][(q, sd)]["median_tpot_ms"] for sd in SEEDS)
        if all((q, sd) in g["plugin-pdl"] for sd in SEEDS):
            c["tpot_plugin_graphs2048"] = st.fmean(g["plugin-pdl"][(q, sd)]["median_tpot_ms"] for sd in SEEDS)
        cells[q] = c
    v: dict = {"status": "complete" if done else "incomplete"}
    if done:
        bad_i = [q for q in RATES if cells[q]["tpot_ratio"] >= 1.0]
        bad_ii = [q for q in RATES if cells[q]["ttft_ratio"] > 1.1]
        v.update({"i": {"met": not bad_i, "not_met_at": bad_i}, "ii": {"met": not bad_ii, "not_met_at": bad_ii}})
    out = {"cells": cells, "verdict": v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
