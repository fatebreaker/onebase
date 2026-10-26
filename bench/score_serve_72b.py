"""SERVE-72B-H200 (docs/REGISTRY.md, registered 2026-10-26; this scorer written before any data): Qwen2.5-72B online on one H200 at
vLLM's default capture, 8 rank-64 tenants, plugin-pdl-lean-fast against stock-tuned (mlsys jobs/pod_runs/serve_72b_h200_pod.sh ->
results/serve_72b_h200_<variant>.jsonl); means over the three starts (request seeds 1-3) of the newest row per start, rates 0.5, 1, 2:
  (i) plugin-pdl-lean-fast's median time per output token below stock-tuned's at every rate
Descriptive: p99 time to first token, output throughput, per-start ratios.
    python3 bench/score_serve_72b.py [--json results/serve_72b_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RATES, SEEDS = ("0.5", "1", "2"), (1, 2, 3)


def rows(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip():
            r = json.loads(line)
            if r.get("rank") == 64 and r.get("n_adapters") == 8:
                out[(str(r["rate"]), r.get("seed", 0))] = r      # later rows replace earlier ones (the warm-up rate 1 is replaced)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    s = rows(LB / "results/serve_72b_h200_stock-tuned.jsonl")
    p = rows(LB / "results/serve_72b_h200_plugin-pdl-lean-fast.jsonl")
    cells, done = {}, True
    for q in RATES:
        if not all((q, sd) in s and (q, sd) in p for sd in SEEDS):
            done = False
            continue
        m = lambda d, k: st.fmean(d[(q, sd)][k] for sd in SEEDS)   # noqa: E731
        c = {"tpot_stock": m(s, "median_tpot_ms"), "tpot_fast": m(p, "median_tpot_ms"), "ttft99_stock": m(s, "p99_ttft_ms"),
             "ttft99_fast": m(p, "p99_ttft_ms"), "out_stock": m(s, "output_throughput"), "out_fast": m(p, "output_throughput")}
        c["tpot_ratio"] = c["tpot_fast"] / c["tpot_stock"]
        c["ttft99_ratio"] = c["ttft99_fast"] / c["ttft99_stock"]
        c["tpot_ratio_per_start"] = [p[(q, sd)]["median_tpot_ms"] / s[(q, sd)]["median_tpot_ms"] for sd in SEEDS]
        cells[q] = c
    out: dict = {"rates": cells, "complete": done}
    if done:
        out["verdict"] = {"i": {"met": all(c["tpot_ratio"] < 1 for c in cells.values()),
                                "not_met_at": [q for q, c in cells.items() if c["tpot_ratio"] >= 1]}}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
