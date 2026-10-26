"""E2E-TP2-VTUNED (docs/REGISTRY.md, registered 2026-10-26; this scorer written before any data): Qwen2.5-72B at TP2 on two H100s
with vLLM's LoRA kernels tuned for the TP2 shard shapes (mock review R3 v42, change 1), the decode path on tiles searched for the same
shards. Offline (mlsys jobs/pod_runs/tp2vt_e2e_pod.sh -> results/e2e_tp2_vtuned.jsonl; means of the three measured starts):
  (i)   W1(plugin-pdl-tiles+r64:n1) < min(W1 of stock-tuned, stock-dual-tuned at 64:1)
  (ii)  W2(plugin-pdl-tiles+r64:n8) > max(W2 of stock-tuned, stock-dual-tuned at 64:8)
Online (tp2vt_serve_pod.sh -> results/serve_tp2_h100_<variant>.jsonl; vLLM's default capture, 8 rank-64 tenants; means over the
three starts of the newest row per start, rates 0.5, 1, 2):
  (iii) plugin-pdl-lean-fast's median time per output token below stock-tuned's at every rate
Descriptive: each path's W1 cost over the engine without LoRA; p99 TTFT and throughput online; per-start ratios.
    python3 bench/score_tp2_vtuned.py [--json results/e2e_tp2_vtuned_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_tp2_vtuned.jsonl"
VLLM = ("stock-tuned", "stock-dual-tuned")
RATES, SEEDS = ("0.5", "1", "2"), (1, 2, 3)


def offline() -> dict:
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
    if all(k in m for k in n1 + ["plugin-pdl-tiles+r64:n1"]):
        best = min(n1, key=lambda k: m[k]["W1_tpot_ms"])
        p = m["plugin-pdl-tiles+r64:n1"]["W1_tpot_ms"]
        out["i"] = {"met": p < m[best]["W1_tpot_ms"], "plugin_ms": p, "vllm_best": best, "vllm_best_ms": m[best]["W1_tpot_ms"],
                    "ratio": p / m[best]["W1_tpot_ms"]}
    n8 = [f"{x}+r64:n8" for x in VLLM]
    if all(k in m for k in n8 + ["plugin-pdl-tiles+r64:n8"]):
        best = max(n8, key=lambda k: m[k]["W2_gen_tok_s"])
        p = m["plugin-pdl-tiles+r64:n8"]["W2_gen_tok_s"]
        out["ii"] = {"met": p > m[best]["W2_gen_tok_s"], "plugin": p, "vllm_best": best, "vllm_best_tok_s": m[best]["W2_gen_tok_s"],
                     "ratio": p / m[best]["W2_gen_tok_s"]}
    if "nolora" in m:
        b0 = m["nolora"]["W1_tpot_ms"]
        out["cost_over_base"] = {k.split("+")[0]: m[k]["W1_tpot_ms"] / b0 - 1 for k in m if k.endswith("+r64:n1")}
    return out


def serve_rows(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip():
            r = json.loads(line)
            if r.get("rank") == 64 and r.get("n_adapters") == 8:
                out[(str(r["rate"]), r.get("seed", 0))] = r      # later rows replace earlier ones (the warm-up rate 1 is replaced)
    return out


def online() -> dict:
    s = serve_rows(LB / "results/serve_tp2_h100_stock-tuned.jsonl")
    p = serve_rows(LB / "results/serve_tp2_h100_plugin-pdl-lean-fast.jsonl")
    cells, done = {}, True
    for q in RATES:
        if not all((q, sd) in s and (q, sd) in p for sd in SEEDS):
            done = False
            continue
        m = lambda d, k: st.fmean(d[(q, sd)][k] for sd in SEEDS)   # noqa: E731
        c = {"tpot_stock": m(s, "median_tpot_ms"), "tpot_fast": m(p, "median_tpot_ms"), "ttft99_stock": m(s, "p99_ttft_ms"),
             "ttft99_fast": m(p, "p99_ttft_ms"), "out_stock": m(s, "output_throughput"), "out_fast": m(p, "output_throughput")}
        c["tpot_ratio"] = c["tpot_fast"] / c["tpot_stock"]
        c["ttft99_ratio"] = c["ttft99_fast"] / c["ttft99_stock"]
        c["tpot_ratio_per_start"] = [p[(q, sd)]["median_tpot_ms"] / s[(q, sd)]["median_tpot_ms"] for sd in SEEDS]
        cells[q] = c
    out: dict = {"rates": cells, "complete": done}
    if done:
        out["iii"] = {"met": all(c["tpot_ratio"] < 1 for c in cells.values()),
                      "not_met_at": [q for q, c in cells.items() if c["tpot_ratio"] >= 1]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {"offline": offline(), "online": online()}
    out["complete_offline"] = "i" in out["offline"] and "ii" in out["offline"]
    out["complete"] = out["complete_offline"] and out["online"]["complete"]
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
