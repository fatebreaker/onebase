"""SERVE-TRACE (docs/REGISTRY.md, registered 2026-10-25 before any data; this scorer written before any data): online serving on a
realistic trace (WildChat first turns, gamma arrivals with burstiness 0.5, SLO goodput at TTFT <= 2 s and TPOT <= 50 ms), 32 rank-64
tenants, Qwen3.5-9B online FP8, CUDA graphs to 2,048 tokens, three server starts (request seeds 1-3) per variant -> rows
results/serve_trace_<gpu>_<variant>.jsonl (bench/serve_bench_t.py). Per (rate, seed) the newest row; means over the three starts.
  (i)  plugin-pdl's median TPOT below stock-tuned's at every measured rate (2, 4, 8, 16; the first pass at 8 is a warm-up, superseded)
  (ii) plugin-pdl's goodput (requests/s meeting both SLOs) at least stock-tuned's at every rate
Descriptive: p99 TTFT, output throughput, completed requests.
    python3 bench/score_serve_trace.py [--gpu h200] [--json results/serve_trace_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RATES = ("2", "4", "8", "16")
SEEDS = (1, 2, 3)


def rows(path: Path) -> dict:
    per: dict = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        k = (str(r.get("rate")), r.get("seed"))
        if k not in per or r.get("time", "") >= per[k].get("time", ""):
            per[k] = r
    return per


def mean(per: dict, rate: str, key: str) -> float | None:
    v = [per[(rate, s)].get(key) for s in SEEDS if (rate, s) in per]
    return sum(v) / len(v) if len(v) == len(SEEDS) and all(x is not None for x in v) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="h200")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    T = rows(LB / f"results/serve_trace_{a.gpu}_stock-tuned.jsonl")
    P = rows(LB / f"results/serve_trace_{a.gpu}_plugin-pdl.jsonl")
    out: dict = {"gpu": a.gpu, "rates": {}}
    ok_i = ok_ii = True
    complete = True
    for q in RATES:
        d = {k: {"stock-tuned": mean(T, q, k), "plugin-pdl": mean(P, q, k)} for k in
             ("median_tpot_ms", "p99_ttft_ms", "request_goodput", "output_throughput", "completed")}
        out["rates"][q] = d
        if any(v is None for v in (d["median_tpot_ms"]["stock-tuned"], d["median_tpot_ms"]["plugin-pdl"],
                                   d["request_goodput"]["stock-tuned"], d["request_goodput"]["plugin-pdl"])):
            complete = False
            continue
        ok_i &= d["median_tpot_ms"]["plugin-pdl"] < d["median_tpot_ms"]["stock-tuned"]
        ok_ii &= d["request_goodput"]["plugin-pdl"] >= d["request_goodput"]["stock-tuned"]
    out["verdict"] = ({"i": {"met": ok_i}, "ii": {"met": ok_ii}} if complete else {"status": "incomplete"})
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
