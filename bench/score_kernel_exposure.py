"""KERNEL-EXPOSURE (docs/REGISTRY.md, registered 2026-10-25 05:36 UTC; this scorer written before any data): the exposure model where it
predicts exposure. Rows: results/pdl_k2_exposure_gh200.jsonl (bench/pdl_k2_bench.py --stages exposure; the newest job). Per case
(one adapter, M = 1, bf16): predicted = max(0, t_shrink_only - t_k2_nolora), measured = t_pdl - t_pdl_noshrink.
Over the cases with predicted >= 2 us (at least 5 of them, else (i) and (ii) are void):
  (i)  measured >= predicted - 1 us in at least 80% of them (the model as a lower bound)
  (ii) Spearman's rho(measured, predicted) >= 0.5
Descriptive: the residual (measured - predicted) per family (model, rank) and for the cases predicting none; the layer's decode-path
time against the faster of vLLM's tuned single- and dual-stream paths per model and rank.
    python3 bench/score_kernel_exposure.py [--json results/kernel_exposure_verdict.json] [--file results/...jsonl]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
NEED = ("t_k2_nolora", "t_shrink_only", "t_pdl", "t_pdl_noshrink")


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = st.fmean(ra), st.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--file", default=str(LB / "results/pdl_k2_exposure_gh200.jsonl"))
    a = ap.parse_args()
    f = Path(a.file)
    rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    rs = [r for r in rs if r.get("stage") == "exposure"]
    if not rs:
        print(json.dumps({"status": "no rows"}))
        return 0
    job = max(str(r.get("job", "")) for r in rs)
    rs = [r for r in rs if str(r.get("job", "")) == job and r.get("n") == 1 and all(isinstance(r.get(k), (int, float)) for k in NEED)]
    cases = [{"model": r["model"], "proj": r["proj"], "r": r["r"], "t_shrink": r["t_shrink_only"], "t_main": r["t_k2_nolora"],
              "pred": max(0.0, r["t_shrink_only"] - r["t_k2_nolora"]), "meas": r["t_pdl"] - r["t_pdl_noshrink"]} for r in rs]
    pos = [c for c in cases if c["pred"] >= 2.0]
    v: dict = {"job": job, "n_cases": len(cases), "n_predicting": len(pos)}
    if len(pos) >= 5:
        lb = [c["meas"] >= c["pred"] - 1.0 for c in pos]
        v["i"] = {"share": sum(lb) / len(lb), "met": sum(lb) / len(lb) >= 0.8}
        rho = spearman([c["meas"] for c in pos], [c["pred"] for c in pos])
        v["ii"] = {"rho": rho, "met": rho >= 0.5}
    else:
        v["status"] = "void: fewer than 5 cases predict 2 us or more"
    fam: dict = {}
    for c in cases:
        fam.setdefault(f"{c['model']} r{c['r']}", []).append(c["meas"] - c["pred"])
    v["residual_by_family"] = {k: {"median": st.median(x), "max": max(x), "n": len(x)} for k, x in fam.items()}
    zero = [c["meas"] for c in cases if c["pred"] == 0.0]
    v["residual_where_none_predicted"] = {"median": st.median(zero), "max": max(zero), "n": len(zero)} if zero else None
    lay: dict = {}
    for r in rs:
        k = f"{r['model']} r{r['r']}"
        d = lay.setdefault(k, {"pdl": 0.0, "vllm": 0.0, "n": 0})
        d["pdl"] += r["t_pdl"]
        d["vllm"] += min(r["t_stock_tuned"], r["t_stock_dual"])
        d["n"] += 1
    v["layer_ratio"] = {k: d["pdl"] / d["vllm"] for k, d in lay.items() if d["n"] == 4}
    v["cases"] = cases
    print(json.dumps(v, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
