"""SERVE-TRACE-DEFAULT-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock reviews R1 v37 missing
4, R3 v35 change 2): SERVE-TRACE at vLLM's default CUDA-graph capture sizes with the plugin's lean base (mlsys jobs/pod_runs/
serve_trace_default_pod.sh -> results/serve_trace_default_h200_<variant>.jsonl); bench/score_serve_trace.py's rule with the variants
stock-tuned and plugin-pdl-lean; means over the three starts.
  (i)  plugin-pdl-lean's median TPOT below stock-tuned's at every measured rate (2, 4, 8, 16)
  (ii) plugin-pdl-lean's goodput (requests/s meeting TTFT <= 2 s and TPOT <= 50 ms) at least stock-tuned's at every rate
Descriptive: p99 TTFT, output throughput.
    python3 bench/score_serve_trace_default.py [--json results/serve_trace_default_verdict.json]"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_serve_trace", LB / "bench/score_serve_trace.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    T = S.rows(LB / "results/serve_trace_default_h200_stock-tuned.jsonl")
    P = S.rows(LB / "results/serve_trace_default_h200_plugin-pdl-lean.jsonl")
    out: dict = {"gpu": "h200", "capture": "vLLM's default", "rates": {}}
    ok_i = ok_ii = True
    complete = True
    for q in S.RATES:
        d = {k: {"stock-tuned": S.mean(T, q, k), "plugin-pdl-lean": S.mean(P, q, k)} for k in
             ("median_tpot_ms", "p99_ttft_ms", "request_goodput", "output_throughput", "completed")}
        out["rates"][q] = d
        if any(v is None for v in (d["median_tpot_ms"]["stock-tuned"], d["median_tpot_ms"]["plugin-pdl-lean"],
                                   d["request_goodput"]["stock-tuned"], d["request_goodput"]["plugin-pdl-lean"])):
            complete = False
            continue
        ok_i &= d["median_tpot_ms"]["plugin-pdl-lean"] < d["median_tpot_ms"]["stock-tuned"]
        ok_ii &= d["request_goodput"]["plugin-pdl-lean"] >= d["request_goodput"]["stock-tuned"]
    out["verdict"] = ({"i": {"met": ok_i}, "ii": {"met": ok_ii}} if complete else {"status": "incomplete"})
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
