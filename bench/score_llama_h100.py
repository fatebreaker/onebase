"""E2E-LLAMA-H100 (docs/REGISTRY.md, registered 2026-10-25 08:31 UTC; this scorer written at 09:38 UTC, after the job had written its
first rows but before any timing in them was read -- see the registry's note): the decode path on Llama-3.1-8B-Instruct with both sides
tuned for Llama's shapes on one H100 (mlsys jobs/pod_runs/e2e_llama_h100_pod.sh -> results/e2e_llama_h100.jsonl); means of the three
measured starts per configuration (labels '#1'-'#3'; '#warm' not kept).
  (i)  W1(plugin-pdl+r64:n1) < min(W1(stock+r64:n1), W1(stock-tuned+r64:n1), W1(stock-dual-tuned+r64:n1))
  (ii) W2(plugin-pdl+r64:n32) > max(W2 of the same three at 64:32)
Descriptive: each path's W1 cost over the engine without LoRA (nolora); plugin-pdl against plugin-pdl-qt (the Qwen3-8B tiles, as
E2E-LLAMA-H200 ran).
    python3 bench/score_llama_h100.py [--json results/llama_h100_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_llama_h100.jsonl"
VLLM = ("stock", "stock-tuned", "stock-dual-tuned")


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
    out: dict = {"means": m}
    n1 = [f"{x}+r64:n1" for x in VLLM]
    if all(k in m for k in n1 + ["plugin-pdl+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p1 = m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p1 < m[best]["W1_tpot_ms"], "plugin_ms": p1, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"],
                    "ratio": p1 / m[best]["W1_tpot_ms"]}
    n32 = [f"{x}+r64:n32" for x in VLLM]
    if all(k in m for k in n32 + ["plugin-pdl+r64:n32"]):
        best = max(n32, key=lambda k: m[k]["W2_gen_tok_s"])
        p2 = m["plugin-pdl+r64:n32"]["W2_gen_tok_s"]
        out["ii"] = {"met": p2 > m[best]["W2_gen_tok_s"], "plugin_tok_s": p2, "vllm_best": best,
                     "vllm_best_tok_s": m[best]["W2_gen_tok_s"], "ratio": p2 / m[best]["W2_gen_tok_s"]}
    if "nolora" in m:
        b0 = m["nolora"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    if "plugin-pdl-qt+r64:n1" in m and "plugin-pdl+r64:n1" in m:
        out["own_tiles_over_qwen_tiles_W1"] = m["plugin-pdl+r64:n1"]["W1_tpot_ms"] / m["plugin-pdl-qt+r64:n1"]["W1_tpot_ms"]
    out["complete"] = "i" in out and "ii" in out and "plugin-pdl-qt+r64:n32" in m
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
