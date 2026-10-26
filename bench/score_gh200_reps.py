"""GH200-E2E-REPS (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v34 M5): Qwen3.5-9B on
the GH200 with vLLM's tuned kernels on its CUTLASS and cuBLASLt bases and in dual-stream mode against the decode path (agent_harness_arm
mlsys/gh200_reps_pod.sh -> results/e2e_gh200_reps.jsonl); means of the measured starts (#warm not kept): three for W1 at 64:1 and the
no-LoRA engines, six for W2 at 64:32.
  (i) W2(plugin-pdl+r64:n32) > max(W2 of stock-tuned, stock-tuned-torch, stock-dual-tuned at 64:32)
Descriptive: W1 at 64:1 against the fastest of the three; each mode's W2 start range; each path's cost over its own base.
    python3 bench/score_gh200_reps.py [--json results/gh200_reps_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_gh200_reps.jsonl"
VLLM = ("stock-tuned", "stock-tuned-torch", "stock-dual-tuned")
NEED = {"W1_tpot_ms": 3, "W2_gen_tok_s": 6}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep != "warm":
                got.setdefault(lab, {})[rep] = r
    m: dict = {}
    for lab, v in got.items():
        m[lab] = {}
        for w in ("W1_tpot_ms", "W2_gen_tok_s"):
            xs = [x[w] for x in v.values() if w in x]
            if xs:
                m[lab][w] = sum(xs) / len(xs)
                m[lab][w + "_n"] = len(xs)
                m[lab][w + "_range"] = [min(xs), max(xs)]
    out: dict = {"means": m}
    n32 = [f"{x}+r64:n32" for x in VLLM]
    ok32 = all(m.get(k, {}).get("W2_gen_tok_s_n", 0) >= NEED["W2_gen_tok_s"] for k in n32 + ["plugin-pdl+r64:n32"])
    if ok32:
        best = max(n32, key=lambda k: m[k]["W2_gen_tok_s"])
        p = m["plugin-pdl+r64:n32"]["W2_gen_tok_s"]
        out["i"] = {"met": p > m[best]["W2_gen_tok_s"], "plugin": p, "vllm_best": best, "vllm_best_tok_s": m[best]["W2_gen_tok_s"],
                    "ratio": p / m[best]["W2_gen_tok_s"]}
    n1 = [f"{x}+r64:n1" for x in VLLM]
    if all(m.get(k, {}).get("W1_tpot_ms_n", 0) >= NEED["W1_tpot_ms"] for k in n1 + ["plugin-pdl+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p = m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["W1_1"] = {"plugin_ms": p, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"], "ratio": p / m[best]["W1_tpot_ms"]}
    out["complete"] = "i" in out and "W1_1" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
