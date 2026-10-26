"""SERVE-L40S-TUNED (docs/REGISTRY.md; this scorer written before any data): online multi-tenant serving on one L40S, the plugin
against vLLM's LoRA kernels tuned for the L40S, from results/serve_l40s_<variant>.jsonl (mlsys jobs/pod_runs/serve_l40s_pod.sh;
bench/serve_bench.py rows, one per rate; per (variant, setting, rate) the newest row, so the warm-up pass at 4 requests/s is
superseded by the measured one). Settings (rank, adapters): (64, 32), (16, 32); rates 1, 2, 4, 6 requests/s.
  (i)  the plugin's median TPOT below stock-tuned's at every rate, in both settings
  (ii) the plugin's p99 TTFT at most 1.2x stock-tuned's at every rate, in both settings
Each met everywhere or not met (the cells where it fails listed); incomplete while a cell lacks a row. Descriptive: every
variant's median and p99 TPOT and TTFT, throughput and KV-cache size; nolora's; plugin-fp4 against the plugin.
    python3 bench/score_serve_l40s.py [--json results/std_bench/serve_l40s_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
VARIANTS = ("nolora", "stock-tuned", "plugin", "plugin-fp4")
SETTINGS = ((64, 32), (16, 32))
RATES = ("1", "2", "4", "6")
KEYS = ("median_tpot_ms", "p99_tpot_ms", "median_ttft_ms", "p99_ttft_ms", "output_throughput", "request_throughput")


def rows(v: str) -> dict:
    f = LB / f"results/serve_l40s_{v}.jsonl"
    out: dict = {}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r.get("rank", 0), r.get("n_adapters", 0), str(r["rate"]))
        if key not in out or r.get("time", "") >= out[key].get("time", ""):
            out[key] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    d = {v: rows(v) for v in VARIANTS}
    table: dict = {}
    for v in VARIANTS:
        for (r, n, rate), row in d[v].items():
            table.setdefault(v, {})[f"{r}:{n}@{rate}"] = {k: row.get(k) for k in KEYS} | {"kv_cache": row.get("kv_cache", "")}
    fails_i, fails_ii, missing = [], [], []
    for r, n in SETTINGS:
        for rate in RATES:
            cell = f"{r}:{n}@{rate}"
            p, t = (d[v].get((r, n, rate)) for v in ("plugin", "stock-tuned"))
            if not (p and t):
                missing.append(cell)
                continue
            best_tpot = t["median_tpot_ms"]
            best_ttft = t["p99_ttft_ms"]
            if not p["median_tpot_ms"] < best_tpot:
                fails_i.append(cell)
            if not p["p99_ttft_ms"] <= 1.2 * best_ttft:
                fails_ii.append(cell)
            table.setdefault("ratios", {})[cell] = {"plugin_tpot_over_tuned": p["median_tpot_ms"] / best_tpot,
                                                    "plugin_p99ttft_over_tuned": p["p99_ttft_ms"] / best_ttft}
    v: dict = {}
    if missing:
        v["i"] = v["ii"] = "incomplete"
    else:
        v["i"] = "met" if not fails_i else f"not met at {', '.join(fails_i)}"
        v["ii"] = "met" if not fails_ii else f"not met at {', '.join(fails_ii)}"
    out = {"table": table, "missing": missing, "verdict": v}
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
