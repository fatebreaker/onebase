"""E2E-TP2-TUNED (docs/REGISTRY.md; post hoc, registered 2026-10-25 after E2E-TP2's data; this scorer written before any of its own):
E2E-TP2's runs again with the decode path on tiles searched for Qwen2.5-72B's TP2 shards (mlsys jobs/pod_runs/tp2tuned_pod.sh ->
results/e2e_tp2_tuned.jsonl); means of the three measured starts.
  (i) W1(plugin-pdl-tiles+r64:n1) < min(W1 of stock, stock-dual at 64:1)
Descriptive: plugin-pdl (default tiles) at 64:1 in the same job; W2 with 8 tenants (64:8) against the better of stock and stock-dual;
each path's W1 cost over the engine without LoRA.
    python3 bench/score_tp2_tuned.py [--json results/e2e_tp2_tuned_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_tp2_tuned.jsonl"
VLLM = ("stock", "stock-dual")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep in ("1", "2", "3"):
                got.setdefault(lab, {})[rep] = r
    m = {k: {w: sum(x[w] for x in v.values()) / len(v) for w in ("W1_tpot_ms", "W2_gen_tok_s")} for k, v in got.items() if len(v) == 3}
    out: dict = {"means": m}
    n1 = [f"{x}+r64:n1" for x in VLLM]
    if all(k in m for k in n1 + ["plugin-pdl-tiles+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p = m["plugin-pdl-tiles+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p < m[best]["W1_tpot_ms"], "plugin_ms": p, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"],
                    "ratio": p / m[best]["W1_tpot_ms"]}
        if "plugin-pdl+r64:n1" in m:
            out["default_tiles_ratio"] = m["plugin-pdl+r64:n1"]["W1_tpot_ms"] / m[best]["W1_tpot_ms"]
    n8 = [f"{x}+r64:n8" for x in VLLM]
    if all(k in m for k in n8 + ["plugin-pdl-tiles+r64:n8"]):
        best = max(n8, key=lambda k: m[k]["W2_gen_tok_s"])
        out["W2_8"] = {"plugin": m["plugin-pdl-tiles+r64:n8"]["W2_gen_tok_s"], "vllm_best": best,
                       "vllm_best_tok_s": m[best]["W2_gen_tok_s"], "ratio": m["plugin-pdl-tiles+r64:n8"]["W2_gen_tok_s"] / m[best]["W2_gen_tok_s"]}
    if "nolora" in m:
        b0 = m["nolora"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    out["complete"] = "i" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
