"""E2E-QWEN3-TUNED-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v34 M5): Qwen3-8B on
one H200 with vLLM's LoRA kernels tuned for Qwen3-8B on this GPU, on its CUTLASS and cuBLASLt bases, against the decode path
(mlsys jobs/pod_runs/floor_tuned_h200_pod.sh -> results/e2e_qwen3tuned_h200.jsonl); means of the three measured starts.
  (i) W1(plugin-pdl+r64:n1) < min(W1 of stock-tuned, stock-tuned-torch, stock-dual-tuned at 64:1)
Descriptive: W2 at 64:32 of the plugin against the best of the three; each path's W1 cost over its own base without LoRA.
    python3 bench/score_qwen3tuned_h200.py [--json results/qwen3tuned_h200_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_qwen3tuned_h200.jsonl"
VLLM = ("stock-tuned", "stock-tuned-torch", "stock-dual-tuned")


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
    if all(k in m for k in n1 + ["plugin-pdl+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        out["i"] = {"met": m["plugin-pdl+r64:n1"]["W1_tpot_ms"] < m[best]["W1_tpot_ms"], "plugin_ms": m["plugin-pdl+r64:n1"]["W1_tpot_ms"],
                    "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"], "ratio": m["plugin-pdl+r64:n1"]["W1_tpot_ms"] / m[best]["W1_tpot_ms"]}
    n32 = [f"{x}+r64:n32" for x in VLLM]
    if all(k in m for k in n32 + ["plugin-pdl+r64:n32"]):
        best = max(n32, key=lambda k: m[k]["W2_gen_tok_s"])
        out["W2_32"] = {"plugin": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"], "vllm_best": best, "vllm_best_tok_s": m[best]["W2_gen_tok_s"],
                        "ratio": m["plugin-pdl+r64:n32"]["W2_gen_tok_s"] / m[best]["W2_gen_tok_s"]}
    out["complete"] = "i" in out and "W2_32" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
