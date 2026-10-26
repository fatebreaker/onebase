"""E2E-72B-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): Qwen2.5-72B-Instruct with vLLM's online
FP8 on one H200; means of the three measured starts. Amendment -TILES (2026-10-25, before any data): scored on the job with the decode
path's tiles searched for the 72B (mlsys jobs/pod_runs/e2e_72b_tiles_h200_pod.sh -> results/e2e_72b_tiles_h200.jsonl); the registered
stream's job (e2e_72b_h200_pod.sh, the decode path on default tiles -> results/e2e_72b_h200.jsonl) is descriptive ("default_tiles").
  (i) W1(plugin-pdl+r64:n1) < min(W1 of stock-tuned, stock-dual-tuned at 64:1)
Descriptive: W2 with 8 tenants (64:8) against the better of the two; each path's W1 cost over the engine without LoRA.
    python3 bench/score_e2e_72b.py [--json results/e2e_72b_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_72b_tiles_h200.jsonl"
F_DEFAULT = LB / "results/e2e_72b_h200.jsonl"
VLLM = ("stock-tuned", "stock-dual-tuned")


def means(f: Path) -> dict:
    got: dict = {}
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep in ("1", "2", "3"):
                got.setdefault(lab, {})[rep] = r
    return {k: {w: sum(x[w] for x in v.values()) / len(v) for w in ("W1_tpot_ms", "W2_gen_tok_s")} for k, v in got.items() if len(v) == 3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    m = means(F)
    out: dict = {"means": m, "default_tiles": means(F_DEFAULT)}
    n1 = [f"{x}+r64:n1" for x in VLLM]
    if all(k in m for k in n1 + ["plugin-pdl+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p = m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p < m[best]["W1_tpot_ms"], "plugin_ms": p, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"],
                    "ratio": p / m[best]["W1_tpot_ms"]}
    n8 = [f"{x}+r64:n8" for x in VLLM]
    if all(k in m for k in n8 + ["plugin-pdl+r64:n8"]):
        best = max(n8, key=lambda k: m[k]["W2_gen_tok_s"])
        out["W2_8"] = {"plugin": m["plugin-pdl+r64:n8"]["W2_gen_tok_s"], "vllm_best": best, "vllm_best_tok_s": m[best]["W2_gen_tok_s"],
                       "ratio": m["plugin-pdl+r64:n8"]["W2_gen_tok_s"] / m[best]["W2_gen_tok_s"]}
    if "nolora-default" in m:
        b0 = m["nolora-default"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    out["complete"] = "i" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
