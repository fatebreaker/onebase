"""TABLE2-MARLIN-L40S (docs/REGISTRY.md, registered 2026-10-26; this scorer written before any data): Qwen3-8B on one L40S with vLLM's
online FP8, vLLM's LoRA kernels tuned for it on the L40S, on vLLM's CUTLASS base (stock-tuned) and its weight-only Marlin base
(stock-tuned-marlin), against the plugin's W8A8 path (plugin) and its weight-only decode path (plugin-w16); bench/overhead.py W1 (ms per
output token, one rank-64 adapter, 64:1) and W2 (generated tokens/s, 32 rank-64 tenants, 64:32); means of the three measured starts
(mlsys jobs/pod_runs/e2e_q3marlin_l40s_pod.sh -> results/e2e_q3marlin_l40s.jsonl).
  (i) the plugin's faster batch-1 path (min W1 of plugin, plugin-w16 at 64:1) below vLLM's faster (min W1 of stock-tuned,
      stock-tuned-marlin at 64:1)
Descriptive: W2 at 64:32 for every mode; each mode's W1 against the engine without LoRA on Marlin's base.
    python3 bench/score_q3marlin_l40s.py [--json results/q3marlin_l40s_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_q3marlin_l40s.jsonl"
VLLM, OURS = ("stock-tuned", "stock-tuned-marlin"), ("plugin", "plugin-w16")


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
    m = {}
    for k, v in got.items():
        if len(v) == 3:
            m[k] = {w: sum(x[w] for x in v.values()) / 3 for w in ("W1_tpot_ms", "W2_gen_tok_s") if all(w in x for x in v.values())}
            m[k]["W1_range"] = [min(x["W1_tpot_ms"] for x in v.values()), max(x["W1_tpot_ms"] for x in v.values())]
    out: dict = {"means": m}
    n1v, n1o = [f"{x}+r64:n1" for x in VLLM], [f"{x}+r64:n1" for x in OURS]
    if all(k in m for k in n1v + n1o):
        bv = min(n1v, key=lambda k: m[k]["W1_tpot_ms"])
        bo = min(n1o, key=lambda k: m[k]["W1_tpot_ms"])
        out["i"] = {"met": m[bo]["W1_tpot_ms"] < m[bv]["W1_tpot_ms"], "vllm_best": bv, "vllm_best_ms": m[bv]["W1_tpot_ms"],
                    "ours_best": bo, "ours_best_ms": m[bo]["W1_tpot_ms"], "ratio": m[bo]["W1_tpot_ms"] / m[bv]["W1_tpot_ms"]}
    out["W2_32"] = {k.split("+")[0]: m[k]["W2_gen_tok_s"] for k in m if k.endswith("+r64:n32") and "W2_gen_tok_s" in m[k]}
    out["complete"] = "i" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
