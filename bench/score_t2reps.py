"""TABLE2-REPS-H200 / -H100 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v37, change 1):
Table 2's Qwen3.5-9B 32-tenant cells on stable baselines: bench/overhead.py's W2 at 64:32, seven starts per mode in one job, pass-major
(#warm not kept) -> results/e2e_t2reps_<gpu>.jsonl; means and ranges of the six measured starts.
  (i) W2(plugin-pdl) > max over vLLM's modes (stock-tuned, stock-tuned-torch, stock-dual-tuned) of their mean W2
Descriptive: each mode's range; the ratio against the best mode's mean and against its best start.
    python3 bench/score_t2reps.py --gpu h200|h100 [--json results/t2reps_<gpu>_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
VLLM = ("stock-tuned", "stock-tuned-torch", "stock-dual-tuned")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True, choices=("h200", "h100"))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    f = LB / f"results/e2e_t2reps_{a.gpu}.jsonl"
    got: dict = {}
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep != "warm" and "W2_gen_tok_s" in r:
                got.setdefault(lab.split("+")[0], []).append(r["W2_gen_tok_s"])
    m = {k: {"mean": sum(v) / len(v), "n": len(v), "range": [min(v), max(v)]} for k, v in got.items()}
    out: dict = {"modes": m}
    if all(m.get(k, {}).get("n", 0) >= 6 for k in VLLM + ("plugin-pdl",)):
        best = max(VLLM, key=lambda k: m[k]["mean"])
        p = m["plugin-pdl"]["mean"]
        out["i"] = {"met": p > m[best]["mean"], "plugin": p, "vllm_best": best, "vllm_best_mean": m[best]["mean"],
                    "ratio": p / m[best]["mean"], "ratio_vs_best_start": p / m[best]["range"][1]}
    out["complete"] = "i" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
