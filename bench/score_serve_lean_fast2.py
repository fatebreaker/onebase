"""SERVE-LEAN-H100-FAST2 (docs/REGISTRY.md, registered 2026-10-26; this scorer written before any data): SERVE-LEAN-H100-FAST's protocol
with plugin-pdl-lean-fast-eager (plugin-pdl-lean-fast plus LOWBIT_EAGER_ABOVE=512: the dispatch times buckets above vLLM's capture bound
in eager calls) against stock-tuned in the same job (mlsys jobs/pod_runs/serve_lean_fast2_h100_pod.sh -> results/
serve_leanfast2_h100_<variant>.jsonl); 32 rank-64 tenants, means over the three starts (request seeds 1-3), SERVE-LEAN-H100's rules:
  (i)  the plugin's median time per output token below stock-tuned's at every rate (2, 4, 8, 16)
  (ii) its p99 time to first token at most 1.1x stock-tuned's at every rate
Descriptive: per-start ratios; against SERVE-LEAN-H100's and SERVE-REPS's rows.
    python3 bench/score_serve_lean_fast2.py [--json results/serve_lean_fast2_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_serve_lean import RES, SEEDS, rows  # noqa: E402

RATES = ("2", "4", "8", "16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--prefix", default="serve_leanfast2", help="the rows' prefix")
    a = ap.parse_args()
    s = rows(RES / f"{a.prefix}_h100_stock-tuned.jsonl")
    p = rows(RES / f"{a.prefix}_h100_plugin-pdl-lean-fast-eager.jsonl")
    old = {v: rows(RES / f"serve_leanreps_h100_{v}.jsonl") for v in ("stock-tuned", "plugin-pdl-lean")}
    g2048 = {v: rows(RES / f"serve_reps_h100_{v}.jsonl") for v in ("stock-tuned", "plugin-pdl")}
    cells, done = {}, True
    for q in RATES:
        if not all((q, sd) in s and (q, sd) in p for sd in SEEDS):
            done = False
            continue
        m = lambda d, k: st.fmean(d[(q, sd)][k] for sd in SEEDS)   # noqa: E731
        c = {"tpot_stock": m(s, "median_tpot_ms"), "tpot_fast": m(p, "median_tpot_ms"),
             "ttft99_stock": m(s, "p99_ttft_ms"), "ttft99_fast": m(p, "p99_ttft_ms")}
        c["tpot_ratio"] = c["tpot_fast"] / c["tpot_stock"]
        c["ttft99_ratio"] = c["ttft99_fast"] / c["ttft99_stock"]
        c["tpot_ratio_per_start"] = [p[(q, sd)]["median_tpot_ms"] / s[(q, sd)]["median_tpot_ms"] for sd in SEEDS]
        for name, d in (("serve_lean_h100", old), ("graphs_2048", g2048)):
            ks = list(d)
            if all((q, sd) in d[k] for k in ks for sd in SEEDS):
                c[name] = {k: m(d[k], "median_tpot_ms") for k in ks}
        cells[q] = c
    out: dict = {"rates": cells, "complete": done}
    if done:
        out["verdict"] = {"i": {"met": all(c["tpot_ratio"] < 1 for c in cells.values()),
                                "not_met_at": [q for q, c in cells.items() if c["tpot_ratio"] >= 1]},
                          "ii": {"met": all(c["ttft99_ratio"] <= 1.1 for c in cells.values()),
                                 "not_met_at": [q for q, c in cells.items() if c["ttft99_ratio"] > 1.1]}}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
