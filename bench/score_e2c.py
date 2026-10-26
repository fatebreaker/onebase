"""ICML-E2C (docs/REGISTRY.md, registered 2026-10-26 before any data): the pilot's samplers with score centering on the engine's
top-32 (--center 32; Marek & Ryabinin 2026, which removes the drift term E_q[R] s-bar) and no importance sampling: arms S8FQC (round
to nearest) and S8FSC (stochastic rounding), seeds 0-2, S8MH's recipe otherwise; from the runs' logs and bf16-engine evaluations.
  (i)   every run completes its 200 steps
  (ii)  S8FSC collapses in at least 2 of 3 runs (the noise account: centering removes the drift but not what SR's per-draw noise
        does; a run collapses as in score_e2pilot_ext.py: late mismatch over 10x early, or final below 10%)
  (iii) every S8FQC run ends above its own pre-RL OlympiadBench+Minerva accuracy on the bf16 engine
Descriptive: late/early mismatch ratios, TV/KL and center_l1 early and late, finals against the uncorrected (S8FQ/S8FS) and TIS
(S8FQT/S8FST) arms, GPU-hours. If (ii) is not met, the drift account explains the pilot's collapse.
    python3 bench/score_e2c.py [--json results/std_bench/e2c_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from score_e2pilot import rows, summary
from score_e2pilot_ext import collapsed

SEEDS = (0, 1, 2)


def center_l1(name: str, lo: int, hi: int) -> float | None:
    xs = [r["center_l1"] for r in rows(name) if lo <= r["step"] <= hi and r.get("center_l1") is not None]
    return st.fmean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: [summary(arm, k) for k in SEEDS] for arm in ("S8FQC", "S8FSC")}
    for xs in R.values():
        for x in xs:
            x["collapsed"] = collapsed(x)
            x["center_l1_first10"] = center_l1(x["run"], 1, 10)
            x["center_l1_last20"] = center_l1(x["run"], 181, 200)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {}
    complete = [x for xs in R.values() for x in xs if x["done"] and x["steps"] >= 200 and x.get("final") is not None]
    if len(complete) == 2 * len(SEEDS):
        v["i"] = {"met": True}
        cs = sum(bool(x["collapsed"]) for x in R["S8FSC"])
        v["ii"] = {"collapses": cs, "met": cs >= 2}
        q = R["S8FQC"]
        v["iii"] = {"final_minus_base": [x["final"] - x["base"] for x in q], "met": all(x["final"] > x["base"] for x in q)}
    else:
        v["status"] = f"incomplete ({len(complete)} of {2 * len(SEEDS)} runs complete and evaluated)"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
