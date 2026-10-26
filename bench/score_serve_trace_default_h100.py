"""SERVE-TRACE-DEFAULT-H100 (docs/REGISTRY.md, registered 2026-10-26; this scorer written before any data; mock review R2 v47
change 4): SERVE-TRACE-DEFAULT-H200's protocol on the H100 (mlsys jobs/pod_runs/serve_trace_default2_pod.sh ->
results/serve_trace_default_h100_<variant>.jsonl); bench/score_serve_trace.py's rule; means over the three starts. The primary
variant is plugin-pdl-lean-fast, the H100's default-capture configuration (SERVE-LEAN-H100-FAST); plugin-pdl-lean-fast-stock32
(SERVE-LEAN-H100-STOCK32's variant) is descriptive.
  (i)  plugin-pdl-lean-fast's median TPOT below stock-tuned's at every measured rate (2, 4, 8, 16)
  (ii) plugin-pdl-lean-fast's goodput (requests/s meeting TTFT <= 2 s and TPOT <= 50 ms) at least stock-tuned's at every rate
Descriptive: p99 TTFT, output throughput, and the same quantities for plugin-pdl-lean-fast-stock32.
    python3 bench/score_serve_trace_default_h100.py [--json results/serve_trace_default_h100_verdict.json]"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_serve_trace", LB / "bench/score_serve_trace.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
VARS = ("stock-tuned", "plugin-pdl-lean-fast", "plugin-pdl-lean-fast-stock32")
PRIMARY = "plugin-pdl-lean-fast"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {v: S.rows(LB / f"results/serve_trace_default_h100_{v}.jsonl") for v in VARS}
    out: dict = {"gpu": "h100", "capture": "vLLM's default", "rates": {}}
    ok_i = ok_ii = True
    complete = True
    for q in S.RATES:
        d = {k: {v: S.mean(R[v], q, k) for v in VARS} for k in
             ("median_tpot_ms", "p99_ttft_ms", "request_goodput", "output_throughput", "completed")}
        out["rates"][q] = d
        if any(v is None for v in (d["median_tpot_ms"]["stock-tuned"], d["median_tpot_ms"][PRIMARY],
                                   d["request_goodput"]["stock-tuned"], d["request_goodput"][PRIMARY])):
            complete = False
            continue
        ok_i &= d["median_tpot_ms"][PRIMARY] < d["median_tpot_ms"]["stock-tuned"]
        ok_ii &= d["request_goodput"][PRIMARY] >= d["request_goodput"]["stock-tuned"]
    out["verdict"] = ({"i": {"met": ok_i}, "ii": {"met": ok_ii}} if complete else {"status": "incomplete"})
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
