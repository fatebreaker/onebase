"""E2E-TP2 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): the plugin under vLLM's tensor parallelism
over two H100s (LOWBIT_TP=1) on a 72B dense model (or Qwen3.6-27B if the 72B did not load), vLLM's online FP8
(mlsys jobs/pod_runs/e2e_tp2_pod.sh -> results/e2e_tp2.jsonl); means of the three measured starts per configuration.
  (i) W1(plugin-pdl+r64:n1) < min(W1(stock+r64:n1), W1(stock-dual+r64:n1))
Descriptive: W2 with 32 tenants (64:32) against the better of stock and stock-dual; each path's W1 cost over nolora; the parity
check (lowbit logs/tp2/parity_*.json) and the fused-layer count.
    python3 bench/score_tp2.py [--json results/tp2_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_tp2.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    got: dict = {}
    model = None
    for line in F.read_text().splitlines() if F.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep in ("1", "2", "3"):
                got.setdefault(lab, {})[rep] = r
                model = r.get("model")
    m = {k: {w: sum(x[w] for x in v.values()) / len(v) for w in ("W1_tpot_ms", "W2_gen_tok_s")} for k, v in got.items() if len(v) == 3}
    out: dict = {"model": model, "means": m}
    n1 = ["stock+r64:n1", "stock-dual+r64:n1"]
    if all(k in m for k in n1 + ["plugin-pdl+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p1 = m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p1 < m[best]["W1_tpot_ms"], "plugin_ms": p1, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"],
                    "ratio": p1 / m[best]["W1_tpot_ms"]}
    n32 = ["stock+r64:n32", "stock-dual+r64:n32"]
    if all(k in m for k in n32 + ["plugin-pdl+r64:n32"]):
        best = max(n32, key=lambda k: m[k]["W2_gen_tok_s"])
        out["W2_32"] = {"plugin": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"], "vllm_best": best,
                        "ratio": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"] / m[best]["W2_gen_tok_s"]}
    if "nolora" in m:
        b0 = m["nolora"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    out["complete"] = "i" in out and "W2_32" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
