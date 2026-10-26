"""KERNEL-H200 (docs/REGISTRY.md; registered 2026-10-24 23:57 UTC before any data). This scorer was written AFTER the data (2026-10-25
02:1x UTC) and implements exactly the registered prediction and descriptives, from results/pdl_k2_h200.jsonl (the newest job):
  (i) with one rank-64 adapter at M = 1, 4, 8 and 16, the decode path's time for a Qwen3-8B layer's four projections (t_pdl summed)
      below the faster of vLLM's tuned single- and dual-stream LoRA paths (min of the summed t_stock_tuned and t_stock_dual) at every M
Descriptive: the adapter's cost over each path's base (t_ref, t_k2_nolora); the exposure model's error per projection (exposed =
t_pdl - t_pdl_noshrink against max(0, t_shrink_only - t_k2_nolora)); the tp and ranks stages' layer ratios.
    python3 bench/score_kernel_h200.py [--json results/kernel_h200_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
PROJS = ("qkv", "o", "gate_up", "down")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rs = [json.loads(x) for x in (LB / "results/pdl_k2_h200.jsonl").read_text().splitlines() if x.strip()]
    job = max(str(r.get("job", "")) for r in rs)
    rs = [r for r in rs if str(r.get("job", "")) == job and r.get("fmt", "bf16") == "bf16" and r.get("n") == 1]

    def layer(stage: str, model: str, r: int, M: int) -> dict | None:
        g = [x for x in rs if x["stage"] == stage and x["model"] == model and x["r"] == r and x["M"] == M]
        if sorted(x["proj"] for x in g) != sorted(PROJS):
            return None
        s = {k: sum(x[k] for x in g) for k in ("t_ref", "t_stock_tuned", "t_stock_dual", "t_pdl", "t_pdl_nopdl")}
        for k in ("t_k2_nolora", "t_pdl_noshrink"):
            s[k] = sum(x[k] for x in g) if all(isinstance(x.get(k), (int, float)) for x in g) else None
        s["vllm_best"] = min(s["t_stock_tuned"], s["t_stock_dual"])
        s["ratio"] = s["t_pdl"] / s["vllm_best"]
        return s

    head = {M: layer("headline", "qwen3_8b", 64, M) for M in (1, 4, 8, 16)}
    v: dict = {"job": job, "gpu": rs[0]["gpu"], "kernel_sha": rs[0]["kernel_sha"]}
    if all(head.values()):
        v["i"] = {"met": all(h["t_pdl"] < h["vllm_best"] for h in head.values()),
                  "layers_us": {M: {"pdl": round(h["t_pdl"], 1), "vllm_best": round(h["vllm_best"], 1), "ratio": round(h["ratio"], 3)}
                                for M, h in head.items()},
                  "pdl_checks_ok": all(x.get("pdl_checks_ok") for x in rs if x["stage"] == "headline")}
        h = head[1]
        v["M1"] = {"base_cutlass": round(h["t_ref"], 1), "base_ours": round(h["t_k2_nolora"], 1),
                   "adapter_cost_pdl": round(h["t_pdl"] - h["t_k2_nolora"], 1),
                   "adapter_cost_in_order": round(h["t_pdl_nopdl"] - h["t_k2_nolora"], 1),
                   "adapter_cost_vllm": round(h["vllm_best"] - h["t_ref"], 1)}
    ex = []
    for x in rs:
        if all(isinstance(x.get(k), (int, float)) for k in ("t_k2_nolora", "t_shrink_only", "t_pdl", "t_pdl_noshrink")) and x["M"] <= 16:
            pred = max(0.0, x["t_shrink_only"] - x["t_k2_nolora"])
            ex.append({"stage": x["stage"], "model": x["model"], "proj": x["proj"], "r": x["r"], "M": x["M"],
                       "ratio": x["t_shrink_only"] / x["t_k2_nolora"], "pred": pred, "meas": x["t_pdl"] - x["t_pdl_noshrink"]})
    err = [abs(e["meas"] - e["pred"]) for e in ex]
    full = [e for e in ex if e["model"] == "qwen3_8b" and e["r"] in (16, 64)]
    v["exposure"] = {"n": len(ex), "median_err": round(st.median(err), 2), "max_err": round(max(err), 2),
                     "full_r16_64_max_exposed": round(max(e["meas"] for e in full), 2),
                     "r128_max_exposed": round(max(e["meas"] for e in ex if e["r"] == 128), 2),
                     "tp_max_exposed": {m: round(max(e["meas"] for e in ex if e["model"] == m), 2)
                                        for m in sorted({e["model"] for e in ex if "_tp" in e["model"]})},
                     "tp_shrink_over_main": [round(min(e["ratio"] for e in ex if "_tp" in e["model"]), 2),
                                             round(max(e["ratio"] for e in ex if "_tp" in e["model"]), 2)]}
    v["tp"] = {}
    for t in (2, 4, 8):
        s = layer("tp", f"qwen3_8b_tp{t}", 64, 1)
        if s:
            v["tp"][t] = {"ratio": round(s["ratio"], 3), "k2_over_cutlass": round(s["t_k2_nolora"] / s["t_ref"], 3),
                          "exposed_layer": round(s["t_pdl"] - s["t_pdl_noshrink"], 1)}
    v["ranks"] = {f"r{r}_M{M}": round(s["ratio"], 3) for r in (16, 128) for M in (1, 16) if (s := layer("ranks", "qwen3_8b", r, M))}
    print(json.dumps(v, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
