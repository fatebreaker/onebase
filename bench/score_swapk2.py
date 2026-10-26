"""HOPPER-SWAP-K2 (docs/REGISTRY.md; descriptive, with a rule), written before its data: from results/swap_k2.jsonl (the newest
job's rows), per batch size M and number of adapters n, one Qwen3-8B decoder layer's four projections summed (microseconds,
rank 64) for vLLM's quantize + CUTLASS without LoRA (ref), vLLM's stock LoRA path, the plugin's hybrid, fused and TMA paths and
the swap path (default tiles; swept when --tune ran). The rule: the swept swap path's per-layer time is below the hybrid
path's at every M <= 16 with one adapter -> an opt-in plugin path and an H100 end-to-end run are registered next.
    python3 bench/score_swapk2.py [--json results/swapk2_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
PROJ = ("qkv", "o", "gate_up", "down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    f = LB / "results/swap_k2.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    if not rows:
        print("no rows")
        return 0
    job = max(r["job"] for r in rows)
    rows = [r for r in rows if r["job"] == job and r["r"] == 64]
    keys = ("t_ref", "t_stock", "t_hybrid", "t_fused", "t_tma", "t_swap", "t_swap_best")
    layer = {}
    for M, n in sorted({(r["M"], r["n"]) for r in rows}):
        rs = [r for r in rows if r["M"] == M and r["n"] == n]
        if sorted(r["proj"] for r in rs) != sorted(PROJ):
            continue
        layer[f"{M}:{n}"] = {k: sum(r[k] for r in rs) if all(k in r for r in rs) else None for k in keys}
        layer[f"{M}:{n}"]["err_swap_max"] = max(r.get("err_swap_best", r.get("err_swap", 0)) for r in rs)
    for k, v in layer.items():
        print(k, {kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    dec = [(k, v) for k, v in layer.items() if int(k.split(":")[1]) == 1 and int(k.split(":")[0]) <= 16]
    best = lambda v: v["t_swap_best"] if v.get("t_swap_best") is not None else v["t_swap"]  # noqa: E731
    met = bool(dec) and all(best(v) < v["t_hybrid"] for _, v in dec)
    verdict = {"job": job, "rule_met": met,
               "decode_one_adapter": {k: {"swap": best(v), "hybrid": v["t_hybrid"], "stock": v["t_stock"], "ref": v["t_ref"],
                                          "swap_over_hybrid": best(v) / v["t_hybrid"], "swap_lora_over_ref": best(v) / v["t_ref"] - 1,
                                          "stock_lora_over_ref": v["t_stock"] / v["t_ref"] - 1} for k, v in dec}}
    print("verdict:", json.dumps(verdict))
    if a.json:
        Path(a.json).write_text(json.dumps({"layer": layer, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
