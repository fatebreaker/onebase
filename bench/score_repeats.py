"""SERVE-REPEATS (docs/REGISTRY.md, registered 2026-10-17 10:00 UTC): three more engine starts of the main L40S warm grids,
Qwen3.5-9B (results/e2e_warm_qwen9_rep{1,2,3}.jsonl) and Qwen3-8B (e2e_warm_qwen3_rep{1,2,3}.jsonl), against the 10-15 start
(e2e_warm_qwen9.jsonl, e2e_warm_qwen3.jsonl). In all three new starts on both models:
  (i)  the plugin's batch-1 cost (W1) for one rank-64 adapter is below the stock path's
  (ii) with 32 rank-64 tenants, NVFP4-stored adapters keep more throughput (W2) than the stock path
Descriptive: mean and range over the three new starts per configuration, the 10-15 start beside them.
    python3 bench/score_repeats.py [--json results/repeats_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "paper/figs"))
import make_figs as mf  # noqa: E402

L40S = "NVIDIA L40S"
MODELS = {"Qwen3.5-9B-text": "qwen9", "Qwen3-8B": "qwen3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out, verdict = {}, {"i": [], "ii": []}
    for model, tag in MODELS.items():
        starts = {}
        for k in ("", "_rep1", "_rep2", "_rep3"):
            d = mf.e2e_platform(L40S, model, "fp8", files=(f"e2e_warm_{tag}{k}.jsonl",))
            if d:
                starts[k or "10-15"] = d
        new = [starts[k] for k in ("_rep1", "_rep2", "_rep3") if k in starts]
        res = {"starts": len(new)}
        if len(new) == 3:
            w1 = [(x[("fused", (64, 1))]["W1"], x[("stock", (64, 1))]["W1"]) for x in new]
            w2 = [(x[("fused-fp4ad", (64, 32))]["W2abs"], x[("stock", (64, 32))]["W2abs"]) for x in new]
            res["i"] = {"plugin_W1": [p for p, _ in w1], "stock_W1": [s for _, s in w1], "met": all(p < s for p, s in w1)}
            res["ii"] = {"fp4_W2abs": [p for p, _ in w2], "stock_W2abs": [s for _, s in w2], "met": all(p > s for p, s in w2)}
            verdict["i"].append(res["i"]["met"])
            verdict["ii"].append(res["ii"]["met"])
            desc = {}
            for key in sorted(set(new[0]) & set(new[1]) & set(new[2])):
                for w in ("W1", "W2"):
                    vals = [x[key][w] for x in new if w in x[key]]
                    if len(vals) == 3:
                        desc[f"{key[0]}|{key[1][0]}x{key[1][1]}|{w}"] = {
                            "mean": st.fmean(vals), "min": min(vals), "max": max(vals),
                            "first": starts.get("10-15", {}).get(key, {}).get(w)}
            res["descriptive"] = desc
        out[model] = res
        print(model, json.dumps({k: v for k, v in res.items() if k != "descriptive"}))
    v = {"i": {"met": len(verdict["i"]) == 2 and all(verdict["i"])}, "ii": {"met": len(verdict["ii"]) == 2 and all(verdict["ii"])}} \
        if len(verdict["i"]) == 2 else {"status": "incomplete"}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"models": out, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
