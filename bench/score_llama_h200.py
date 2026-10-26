"""E2E-LLAMA-H200 (docs/REGISTRY.md, registered 2026-10-25 04:21 UTC; this scorer written before any data): the decode path on
Llama-3.1-8B-Instruct (dense) with vLLM's online FP8 on one H200 (mlsys jobs/pod_runs/e2e_llama_h200_pod.sh ->
results/e2e_llama_h200.jsonl); means of the three measured starts per configuration (labels '#1'-'#3'; '#warm' not kept).
  (i) W1(plugin-pdl+r64:n1) < min(W1(stock+r64:n1), W1(stock-dual+r64:n1))
Descriptive: each path's cost over the engine without LoRA (nolora); W2 at 64:32 against the better of stock and stock-dual.
    python3 bench/score_llama_h200.py [--json results/llama_h200_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_llama_h200.jsonl"
LABELS = ("nolora", "stock+r64:n1", "stock-dual+r64:n1", "plugin-pdl+r64:n1", "stock+r64:n32", "stock-dual+r64:n32", "plugin-pdl+r64:n32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        if not line.strip():
            continue
        r = json.loads(line)
        lab, _, rep = r["label"].rpartition("#")
        if rep in ("1", "2", "3"):
            got.setdefault(lab, {})[rep] = r
    m = {k: {w: sum(x[w] for x in v.values()) / len(v) for w in ("W1_tpot_ms", "W2_gen_tok_s")} for k, v in got.items() if len(v) == 3}
    out: dict = {"means": m, "complete": all(k in m for k in LABELS)}
    if out["complete"]:
        best1 = min(m["stock+r64:n1"]["W1_tpot_ms"], m["stock-dual+r64:n1"]["W1_tpot_ms"])
        p1 = m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p1 < best1, "plugin_ms": p1, "vllm_best_ms": best1, "ratio": p1 / best1}
        b0 = m["nolora"]["W1_tpot_ms"]
        out["cost_over_base"] = {k: m[f"{k}+r64:n1"]["W1_tpot_ms"] / b0 - 1 for k in ("stock", "stock-dual", "plugin-pdl")}
        w_best = max(m["stock+r64:n32"]["W2_gen_tok_s"], m["stock-dual+r64:n32"]["W2_gen_tok_s"])
        out["W2_32"] = {"plugin": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"], "vllm_best": w_best,
                        "ratio": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"] / w_best}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
