"""ACC-QWEN (docs/REGISTRY.md, SERVE-QWEN amendment; descriptive): the exact adapter on vLLM's online FP8 base served by
vLLM's LoRA path (fused0) and by the plugin (fused1), per adapter and benchmark: accuracies, plugin minus stock, the
discordant pairs and McNemar's exact p (mlsys analysis/read_retention.py).
Files: results/std_bench/<run>_fp8eng_fused<F>_<gpu>/fp8eng_exact/<bench>.jsonl.
    python3 bench/score_fp8eng.py [--runs qwen9_dapo_S8G_s0 qwen9_dapo_S0G_s2] [--gpu l40s] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["qwen9_dapo_S8G_s0", "qwen9_dapo_S0G_s2"])
    ap.add_argument("--gpu", default="l40s")
    ap.add_argument("--benches", nargs="+", default=["math500", "olympiad", "minerva"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for run in a.runs:
        for b in a.benches:
            s, p = (load(RES / f"{run}_fp8eng_fused{f}_{a.gpu}/fp8eng_exact/{b}.jsonl") for f in (0, 1))
            if s is None or p is None:
                print(f"{run} {b}: missing")
                continue
            ids = sorted(set(s) & set(p))
            stock_wins, plugin_wins, pv = rr.mcnemar([s[i] for i in ids], [p[i] for i in ids])
            e = {"n": len(ids), "stock": 100 * sum(s[i] for i in ids) / len(ids), "plugin": 100 * sum(p[i] for i in ids) / len(ids),
                 "stock_only": stock_wins, "plugin_only": plugin_wins, "p": pv}
            e["diff"] = e["plugin"] - e["stock"]
            out[f"{run}:{b}"] = e
            print(f"{run} {b}: plugin {e['plugin']:.1f} vs stock {e['stock']:.1f} ({e['diff']:+.1f}; discordant "
                  f"{plugin_wins} vs {stock_wins}, p = {pv:.2f}), n {len(ids)}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
