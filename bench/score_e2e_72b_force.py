"""E2E-72B-H200-FORCE (docs/REGISTRY.md; post hoc, registered 2026-10-26; this scorer written before any data): Qwen2.5-72B on one H200,
the decode path forced for buckets up to 32 tokens (plugin-pdl-force, LOWBIT_PDL_FORCE=1) beside the timed dispatch (plugin-pdl), against
vLLM's tuned kernels (stock-tuned, stock-dual-tuned) in the same job (results/e2e_72b_force_h200.jsonl); means of the three measured starts.
  (i) W1(plugin-pdl-force+r64:n1) < W1(plugin-pdl+r64:n1)   (forcing the decode path is faster than the timed choice at batch 1)
Descriptive: each plugin mode against the faster vLLM mode at 64:1 and 64:8; each mode's cost over the engine without LoRA.
    python3 bench/score_e2e_72b_force.py [--json results/e2e_72b_force_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_72b_force_h200.jsonl"
VLLM = ("stock-tuned", "stock-dual-tuned")


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
    m = {k: {w: sum(x[w] for x in v.values()) / 3 for w in ("W1_tpot_ms", "W2_gen_tok_s") if all(w in x for x in v.values())}
         for k, v in got.items() if len(v) == 3}
    out: dict = {"means": m}
    if all(f"{x}+r64:n1" in m for x in ("plugin-pdl", "plugin-pdl-force")):
        f, d = m["plugin-pdl-force+r64:n1"]["W1_tpot_ms"], m["plugin-pdl+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": f < d, "force_ms": f, "timed_ms": d, "ratio": f / d}
    for s, w, best in (("n1", "W1_tpot_ms", min), ("n8", "W2_gen_tok_s", max)):
        vs = [f"{x}+r64:{s}" for x in VLLM if f"{x}+r64:{s}" in m and w in m[f"{x}+r64:{s}"]]
        if vs:
            bv = best(vs, key=lambda k: m[k][w])
            out[f"vs_vllm_{s}"] = {"vllm_best": bv, "vllm": m[bv][w],
                                   **{p: m[f"{p}+r64:{s}"][w] / m[bv][w] for p in ("plugin-pdl", "plugin-pdl-force") if f"{p}+r64:{s}" in m}}
    if "nolora-default" in m:
        b0 = m["nolora-default"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    out["complete"] = "i" in out and "vs_vllm_n8" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
