"""SERVE-POOL-TUNED (docs/REGISTRY.md, registered before any data; this scorer written before any data; mock review R2 v46): the
128-adapter pool of SERVE-POOL-TRUNC (128 random rank-16 adapters in 128 bf16 slots, one L40S) with vLLM's LoRA kernels tuned for
Qwen3.5-9B on the L40S (stock-tuned) against ours (fused) in the same jobs (jobs/serve_pool_tuned.sbatch ->
results/serve_pool_tuned.jsonl; each configuration's newest row per seed and rate, so the second "inf" pass replaces the warm-up).
  (i) at 4 requests/s ours has a lower p99 time to first token than stock-tuned in every seed
Descriptive: p99 TTFT, median TPOT and output throughput per configuration and rate (mean and range over seeds 0-2), and stock-tuned
against SERVE-POOL-TRUNC's stock path with vLLM's default kernels.
    python3 bench/score_pool_tuned.py [--json results/pool_tuned_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RATES, SEEDS = ("2", "4", "6", "inf"), (0, 1, 2)


def load(path: Path) -> dict:
    out: dict = {}
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    for r in sorted(rows, key=lambda r: r.get("time", "")):
        out[(r["mode"], int(r.get("max_loras") or 0), int(r.get("rank") or 0), int(r["seed"]), str(r["rate"]))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    t = load(LB / "results/serve_pool_tuned.jsonl")
    d = load(LB / "results/serve_pool_trunc.jsonl")
    cfg = {"stock-tuned": (t, "stock"), "fused": (t, "fused"), "stock-default": (d, "stock")}
    seeds = [s for s in SEEDS if all((m, 128, 16, s, q) in src for src, m in cfg.values() for q in RATES)]
    desc: dict = {}
    for name, (src, m) in cfg.items():
        for q in RATES:
            rs = [src[(m, 128, 16, s, q)] for s in seeds]
            if rs:
                desc[f"{name}@{q}"] = {k: {"mean": sum(r[k] for r in rs) / len(rs), "min": min(r[k] for r in rs), "max": max(r[k] for r in rs)}
                                       for k in ("p99_ttft_ms", "median_tpot_ms", "output_throughput") if all(k in r for r in rs)}
    out: dict = {"seeds": seeds, "desc": desc}
    if len(seeds) == len(SEEDS):
        per = {s: (t[("fused", 128, 16, s, "4")]["p99_ttft_ms"], t[("stock", 128, 16, s, "4")]["p99_ttft_ms"]) for s in seeds}
        out["i"] = {"per_seed": per, "met": all(o < v for o, v in per.values())}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
