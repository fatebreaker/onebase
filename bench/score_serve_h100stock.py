"""SERVE-H100-PDL-STOCK (docs/REGISTRY.md; this scorer written before any data): SERVE-H100-PDL's predictions applied to the plugin
with the decode path for decode buckets and vLLM's own LoRA kernels above them (plugin-pdl-stock: LOWBIT_STOCK_ABOVE=32), against
SERVE-H100-PDL's stock-tuned and stock-dual rows (same H100, a separate job). Settings (64, 32), (16, 32); rates 2, 4, 8, 16.
  (i)  plugin-pdl-stock's median TPOT below the better of stock-tuned's and stock-dual's at every rate, in both settings
  (ii) its p99 TTFT at most 1.2x the better of the two at every rate, in both settings
Descriptive: against plugin-pdl (SERVE-H100-PDL) and the plugin without the decode path (SERVE-H100-PDL-DIAG).
    python3 bench/score_serve_h100stock.py [--json results/std_bench/serve_h100stock_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
VARIANTS = ("nolora", "stock-tuned", "stock-dual", "plugin-pdl", "plugin-pdl-fp4", "plugin", "plugin-pdl-k", "plugin-pdl-stock")
SETTINGS = ((64, 32), (16, 32))
RATES = ("2", "4", "8", "16")
KEYS = ("median_tpot_ms", "p99_tpot_ms", "median_ttft_ms", "p99_ttft_ms", "output_throughput", "request_throughput")


def rows(v: str) -> dict:
    f = LB / f"results/serve_h100pdl_{v}.jsonl"
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
            p, t, du = (d[v].get((r, n, rate)) for v in ("plugin-pdl-stock", "stock-tuned", "stock-dual"))
            if not (p and t and du):
                missing.append(cell)
                continue
            best_tpot = min(t["median_tpot_ms"], du["median_tpot_ms"])
            best_ttft = min(t["p99_ttft_ms"], du["p99_ttft_ms"])
            if not p["median_tpot_ms"] < best_tpot:
                fails_i.append(cell)
            if not p["p99_ttft_ms"] <= 1.2 * best_ttft:
                fails_ii.append(cell)
            table.setdefault("ratios", {})[cell] = {"pdl_tpot_over_best": p["median_tpot_ms"] / best_tpot,
                                                    "pdl_p99ttft_over_best": p["p99_ttft_ms"] / best_ttft}
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
