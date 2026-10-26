"""BLACKWELL-PDL-K2 (docs/REGISTRY.md, registered before any data): its three predictions from
results/pdl_fp4.jsonl (bench/pdl_fp4_bench.py), one decoder layer of our NVFP4 Qwen3-8B on the Blackwell slice, one rank-64 bf16 adapter, M in {1, 4, 8, 16}:
  (i)   the pdl layer's time (layer_pdl, one graph) below vLLM's LoRA path tuned for the Blackwell slice's (layer_stock_tuned) at every M
  (ii)  the pdl layer's time at most 1.15x CUTLASS alone (layer_ref) at every M
  (iii) the sum of pdl's four projection times below the sum of stock_dual's (vLLM's dual-stream LoRA) at every M
Also every output check and the determinism and quantization checks of the rows used.
    python3 bench/score_pdlfp4.py [--json results/std_bench/pdlfp4_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
MS = (1, 4, 8, 16)
PROJS = ("qkv", "o", "gate_up", "down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    f = LB / "results/pdl_fp4.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    lay = {r["M"]: r for r in rows if r.get("stage") == "layer" and r.get("model") == "qwen3_8b" and r.get("n") == 1 and r.get("r") == 64}
    proj = {}
    for r in rows:
        if r.get("stage") in ("headline",) and r.get("model") == "qwen3_8b" and r.get("r") == 64 and r.get("n") == 1 and r.get("fmt", "bf16") == "bf16":
            proj[(r["M"], r["proj"])] = r
    v, per = {}, {}
    for M in MS:
        L = lay.get(M)
        P = [proj.get((M, p)) for p in PROJS]
        if not L or not all(P):
            v["status"] = f"incomplete (M = {M})"
            break
        per[M] = {"layer_pdl": L["t_layer_pdl"], "layer_ref": L["t_layer_ref"], "layer_stock_tuned": L["t_layer_stock_tuned"],
                  "sum_pdl": sum(p["t_pdl"] for p in P), "sum_stock_dual": sum(p["t_stock_dual"] for p in P),
                  "checks_ok": all(p.get("checks_ok") and p.get("pdl_checks_ok") for p in P),
                  "deterministic": all(p.get("pdl_deterministic") for p in P)}
    if "status" not in v:
        v["i"] = {"ratios": {M: x["layer_pdl"] / x["layer_stock_tuned"] for M, x in per.items()}}
        v["i"]["met"] = all(r < 1 for r in v["i"]["ratios"].values())
        v["ii"] = {"ratios": {M: x["layer_pdl"] / x["layer_ref"] for M, x in per.items()}}
        v["ii"]["met"] = all(r <= 1.15 for r in v["ii"]["ratios"].values())
        v["iii"] = {"ratios": {M: x["sum_pdl"] / x["sum_stock_dual"] for M, x in per.items()}}
        v["iii"]["met"] = all(r < 1 for r in v["iii"]["ratios"].values())
        v["checks_ok"] = all(x["checks_ok"] for x in per.values())
        v["deterministic"] = all(x["deterministic"] for x in per.values())
    out = {"per_M": per, "verdict": v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
