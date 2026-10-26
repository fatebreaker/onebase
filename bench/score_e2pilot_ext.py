"""ICML-E2-PILOT -EXT234 (docs/REGISTRY.md, registered 2026-10-26 before any data of seeds 2-4): the pilot's arms (S8FQ to nearest,
S8FS stochastically, no correction), seeds 0-4, from the runs' logs and bf16-engine evaluations (score_e2pilot.py's readers).
Per run, the pilot's rules:
  (i)   every run completes its 200 steps
  (ii)  S8FQ: mean mismatch over steps 181-200 at least 0.015 and at least 1.2x its mean over steps 1-10 (drift)
  (iii) S8FS: mean mismatch over steps 181-200 at most 1.1x its mean over steps 1-10 (no drift)
Pooled over seeds 0-4:
  (iv)  S8FS collapses more often than S8FQ: Fisher's exact test, one-sided, p < 0.05. A run collapses when its mean mismatch
        over steps 181-200 exceeds 10x its mean over steps 1-10, or its final OlympiadBench+Minerva accuracy on the bf16 engine
        is below 10%.
Descriptive: collapse counts, finals against S0TF and S8MH, TV/KL early and late, GPU-hours.
    python3 bench/score_e2pilot_ext.py [--json results/std_bench/e2pilot_ext_verdict.json]"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from score_e2pilot import summary

SEEDS = (0, 1, 2, 3, 4)


def collapsed(x: dict) -> bool | None:
    if x.get("mismatch_last20") is None or x.get("final") is None:
        return None
    return x["mismatch_last20"] > 10 * x["mismatch_first10"] or x["final"] < 10.0


def fisher_greater(a: int, n1: int, c: int, n2: int) -> float:
    """One-sided Fisher exact p that group 1 (a of n1) has the higher rate than group 2 (c of n2)."""
    k = a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: [summary(arm, k) for k in SEEDS] for arm in ("S8FQ", "S8FS")}
    for xs in R.values():
        for x in xs:
            x["collapsed"] = collapsed(x)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {}
    complete = [x for xs in R.values() for x in xs if x["done"] and x["steps"] >= 200 and x.get("final") is not None]
    if len(complete) == 2 * len(SEEDS):
        v["i"] = {"met": True}
        q, s = R["S8FQ"], R["S8FS"]
        v["ii"] = {"ratio": [x["mismatch_last20"] / x["mismatch_first10"] for x in q],
                   "met_per_run": [x["mismatch_last20"] >= 0.015 and x["mismatch_last20"] >= 1.2 * x["mismatch_first10"] for x in q]}
        v["iii"] = {"ratio": [x["mismatch_last20"] / x["mismatch_first10"] for x in s],
                    "met_per_run": [x["mismatch_last20"] <= 1.1 * x["mismatch_first10"] for x in s]}
        cs, cq = sum(bool(x["collapsed"]) for x in s), sum(bool(x["collapsed"]) for x in q)
        p = fisher_greater(cs, len(s), cq, len(q))
        v["iv"] = {"collapses_S8FS": cs, "collapses_S8FQ": cq, "p_one_sided": p, "met": p < 0.05}
    else:
        v["status"] = f"incomplete ({len(complete)} of {2 * len(SEEDS)} runs complete and evaluated)"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
