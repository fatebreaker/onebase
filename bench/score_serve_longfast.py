"""SERVE-LONGPROMPT-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): 2,048-token prompts at vLLM's
default capture on one H200, plugin-pdl-lean-fast against stock-tuned (mlsys jobs/pod_runs/serve_longprompt_fast_h200_pod.sh ->
results/serve_longfast_h200_<variant>.jsonl); 32 rank-64 tenants, means over the three starts (request seeds 1-3), rates 1, 2, 4:
  (i) plugin-pdl-lean-fast's median time per output token below stock-tuned's at every rate
Descriptive: p99 time to first token (SERVE-LONGPROMPT on the H100 found it 1.18-1.32x), throughput, per-start ratios.
SERVE-LONGPROMPT-H100-FAST (registered 2026-10-25 before any data; --gpu h100): the same on MLSys's H100.
    python3 bench/score_serve_longfast.py [--gpu h200|h100] [--json results/serve_longfast_<gpu>_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_serve_lean import RES, SEEDS, rows  # noqa: E402

RATES = ("1", "2", "4")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--gpu", default="h200", help="SERVE-LONGPROMPT-H100-FAST: h100")
    a = ap.parse_args()
    s = rows(RES / f"serve_longfast_{a.gpu}_stock-tuned.jsonl")
    p = rows(RES / f"serve_longfast_{a.gpu}_plugin-pdl-lean-fast.jsonl")
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
