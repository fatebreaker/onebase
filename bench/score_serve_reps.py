"""SERVE-REPS (docs/REGISTRY.md, registered before any data): SERVE-GRAPHS's online protocol (CUDA graphs up to 2,048 tokens for
every configuration) with three server starts per configuration, request seeds 1-3 shared by every variant, and 1,000 requests
per rate (mlsys jobs/pod_runs/serve_reps_pod.sh -> results/serve_reps_<gpu>_<variant>.jsonl). Per GPU, 32 rank-64 and 32 rank-16
tenants, the mean over the three seeds at each measured rate (L40S 1, 2, 4, 6 requests/s; H100 2, 4, 8, 16; the list's first rate
warms the server and is overwritten by its repeat):
  (i)   the plugin's (L40S: plugin; H100: plugin-pdl) median time per output token below stock-tuned's at every rate, both settings
  (ii)  the plugin's p99 time to first token at most 1.1x stock-tuned's at every rate, both settings
Descriptive: NVFP4-stored adapters (L40S plugin-fp4) against stock-tuned; per seed, the plugin's TPOT over stock-tuned's (paired).
A GPU's verdict needs all three seeds of every configuration.
    python3 bench/score_serve_reps.py [--json results/serve_reps_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
GPUS = {"l40s": ("plugin", ("1", "2", "4", "6"), ("stock-tuned", "plugin", "plugin-fp4")),
        "h100": ("plugin-pdl", ("2", "4", "8", "16"), ("stock-tuned", "plugin-pdl"))}
SETTINGS = ((64, 32), (16, 32))
SEEDS = (1, 2, 3)
# Amendment (2026-10-23): the L40S batch starts of 10-23 shared one node, /tmp dir and port, so every client drove one server; they
# are void and the L40S rerun (mlsys serve_reps2_pod.sh) writes results/serve_reps2_l40s_<variant>.jsonl.
PRE = {"l40s": "serve_reps2", "h100": "serve_reps"}


def rows(path: Path) -> dict:
    """{(rank, n, rate, seed): row}, the newest row per key."""
    out = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip():
            r = json.loads(line)
            out[(r["rank"], r["n_adapters"], str(r["rate"]), r.get("seed", 0))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for g, (ours, rates, variants) in GPUS.items():
        R = {v: rows(LB / f"results/{PRE[g]}_{g}_{v}.jsonl") for v in variants}
        need = [(v, r, n, q, s) for v in variants for (r, n) in SETTINGS for q in rates for s in SEEDS]
        miss = [k for k in need if (k[1], k[2], k[3], k[4]) not in R[k[0]]]
        if miss:
            res[g] = {"status": "incomplete", "missing": len(miss), "of": len(need)}
            print(g, "incomplete:", len(miss), "of", len(need))
            continue
        mean = lambda v, r, n, q, f: sum(R[v][(r, n, q, s)][f] for s in SEEDS) / len(SEEDS)  # noqa: E731
        per = {}
        for (r, n) in SETTINGS:
            for q in rates:
                st_t, ou_t = mean("stock-tuned", r, n, q, "median_tpot_ms"), mean(ours, r, n, q, "median_tpot_ms")
                st_f, ou_f = mean("stock-tuned", r, n, q, "p99_ttft_ms"), mean(ours, r, n, q, "p99_ttft_ms")
                e = {"tpot_stock": st_t, "tpot_ours": ou_t, "tpot_ratio": ou_t / st_t, "ttft_stock": st_f, "ttft_ours": ou_f,
                     "ttft_ratio": ou_f / st_f,
                     "tpot_ratio_by_seed": [R[ours][(r, n, q, s)]["median_tpot_ms"] / R["stock-tuned"][(r, n, q, s)]["median_tpot_ms"]
                                            for s in SEEDS]}
                if "plugin-fp4" in variants:
                    e["tpot_ratio_fp4"] = mean("plugin-fp4", r, n, q, "median_tpot_ms") / st_t
                    e["ttft_ratio_fp4"] = mean("plugin-fp4", r, n, q, "p99_ttft_ms") / st_f
                per[f"{r}:{n}@{q}"] = e
        v = {"i": {"met": all(e["tpot_ratio"] < 1 for e in per.values()),
                   "not_met_at": [k for k, e in per.items() if not e["tpot_ratio"] < 1]},
             "ii": {"met": all(e["ttft_ratio"] <= 1.1 for e in per.values()),
                    "not_met_at": [k for k, e in per.items() if not e["ttft_ratio"] <= 1.1]}}
        res[g] = {"cells": per, "verdict": v}
        for k, e in per.items():
            print(g, k, json.dumps({x: (round(y, 3) if isinstance(y, float) else [round(z, 3) for z in y] if isinstance(y, list) else y)
                                    for x, y in e.items()}))
        print(g, "verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
