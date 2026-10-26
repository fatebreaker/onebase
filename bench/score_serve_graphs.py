"""SERVE-GRAPHS (docs/REGISTRY.md; this scorer written before any data): online multi-tenant serving with CUDA graphs captured up to
2,048 tokens for every configuration, the plugin against vLLM's tuned LoRA kernels, from results/serve_graphs_<gpu>_<variant>.jsonl
(mlsys jobs/pod_runs/serve_graphs_pod.sh; bench/serve_bench_x.py rows; per (variant, setting, rate) the newest row, so the warm-up
pass is superseded). Plugin variant: plugin-pdl on the H100, plugin on the L40S. Settings (64, 32), (16, 32).
  (i)  the plugin's median TPOT below stock-tuned's at every rate, in both settings
  (ii) the plugin's p99 TTFT at most 1.1x stock-tuned's at every rate, in both settings
Descriptive: each configuration against its default-capture rows (SERVE-H100-PDL, SERVE-L40S-TUNED).
    python3 bench/score_serve_graphs.py --gpu h100|l40s [--json results/std_bench/serve_graphs_<gpu>_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
KEYS = ("median_tpot_ms", "p99_tpot_ms", "median_ttft_ms", "p99_ttft_ms", "output_throughput")


def rows(path: Path) -> dict:
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            key = (r.get("rank", 0), r.get("n_adapters", 0), str(r["rate"]))
            if key not in out or r.get("time", "") >= out[key].get("time", ""):
                out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=("h100", "l40s"), required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    plug = "plugin-pdl" if a.gpu == "h100" else "plugin"
    rates = ("2", "4", "8", "16") if a.gpu == "h100" else ("1", "2", "4", "6")
    base = {"h100": "serve_h100pdl", "l40s": "serve_l40s"}[a.gpu]
    g = {v: rows(LB / f"results/serve_graphs_{a.gpu}_{v}.jsonl") for v in ("stock-tuned", plug)}
    d = {v: rows(LB / f"results/{base}_{v}.jsonl") for v in ("stock-tuned", plug)}
    cells, fi, fii, missing = {}, [], [], []
    for r_, n in ((64, 32), (16, 32)):
        for q in rates:
            c = f"{r_}:{n}@{q}"
            p, t = g[plug].get((r_, n, q)), g["stock-tuned"].get((r_, n, q))
            if not (p and t):
                missing.append(c)
                continue
            e = {"plugin": {k: p.get(k) for k in KEYS}, "stock_tuned": {k: t.get(k) for k in KEYS},
                 "tpot_ratio": p["median_tpot_ms"] / t["median_tpot_ms"], "p99ttft_ratio": p["p99_ttft_ms"] / t["p99_ttft_ms"]}
            for v, lab in ((plug, "plugin_default"), ("stock-tuned", "stock_tuned_default")):
                x = d[v].get((r_, n, q))
                if x:
                    e[lab] = {k: x.get(k) for k in KEYS}
            cells[c] = e
            if not e["tpot_ratio"] < 1.0:
                fi.append(c)
            if not e["p99ttft_ratio"] <= 1.1:
                fii.append(c)
    v = {"i": "incomplete" if missing else ("met" if not fi else f"not met at {', '.join(fi)}"),
         "ii": "incomplete" if missing else ("met" if not fii else f"not met at {', '.join(fii)}")}
    out = {"gpu": a.gpu, "cells": cells, "missing": missing, "verdict": v}
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
