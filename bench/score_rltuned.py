"""RL-FP8-QMATH-TUNED (docs/REGISTRY.md, registered 2026-10-16 15:16 UTC): the RL comparison against a tuned bf16 baseline,
a full-length unmatched trainer and the FP8 engine without our kernels.
  (i)   S0T's mean generation time per step within 0.8-1.25x S8G's (means over seeds 0-2)
  (ii)  S9P's GPU-hours per run at most 0.8x S0T's mean (S9B descriptive)
  (iii) final accuracy on OlympiadBench+Minerva, S8G minus S4F over seeds 0-2, Welch 95% interval within +-5
  (iv)  S8L's mean generation time per step at most 0.9x S8LV's (Qwen3-4B, L40S)
Descriptive: S0T's accuracy against S0G's; GPU-hours per run of every arm; S0T's sleep and wake times.
Arms' logs and evaluations as bench/score_rlqmath.py and score_rlq4l.py read them (S0T, S4F: qwen9_dapo_<arm>_s<k>;
S8LV: qwen4_dapo_S8LV_s<k>, 100 steps).
    python3 bench/score_rltuned.py [--json results/std_bench/rltuned_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlq4l as q4  # noqa: E402
import score_rlqmath as qm  # noqa: E402
from score_rlmulti import welch  # noqa: E402

qm.EV.update({"S0T": ("bf16_base", "bf16_exact"), "S4F": ("fp8eng_base", "fp8eng_exact")})
q4.EV.update({"S8LV": ("fp8eng_base", "fp8eng_exact")})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    seeds = (0, 1, 2)
    R = {arm: [x for x in (qm.summary(arm, s) for s in seeds) if x] for arm in ("S0G", "S8G", "S0T", "S4F", "S9P", "S9B")}
    for arm in ("S0T", "S4F"):
        for x in R[arm]:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    L4 = {arm: [x for x in (q4.summary(arm, s) for s in seeds) if x] for arm in ("S8L", "S8LV")}
    for x in L4["S8LV"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    mean = qm.mean
    v = {}
    if len(R["S0T"]) == 3 and len(R["S8G"]) == 3:
        t0t, t8 = mean([x["t_gen_mean"] for x in R["S0T"]]), mean([x["t_gen_mean"] for x in R["S8G"]])
        v["i"] = {"S0T": t0t, "S8G": t8, "ratio": t0t / t8, "met": 0.8 <= t0t / t8 <= 1.25}
        h0t = mean([x["gpu_hours"] for x in R["S0T"]])
        v["gpu_hours"] = {arm: mean([x["gpu_hours"] for x in R[arm]]) for arm in R if R[arm]}
        for arm in ("S9P", "S9B"):
            if len(R[arm]) == 3:
                h = mean([x["gpu_hours"] for x in R[arm]])
                v[f"ii_{arm}"] = {"gpu_hours": h, "S0T": h0t, "ratio": h / h0t, "met": h <= 0.8 * h0t}
        acc0 = [x["hard_final"] for x in R["S0G"]]
        acct = [x["hard_final"] for x in R["S0T"] if x.get("hard_final") is not None]
        if len(acct) >= 2 and len(acc0) >= 2:
            v["S0T_vs_S0G"] = welch(acct, acc0)
    if len(R["S4F"]) == 3 and len(R["S8G"]) == 3:
        w = welch([x["hard_final"] for x in R["S8G"]], [x["hard_final"] for x in R["S4F"]])
        v["iii"] = {**w, "met": w["verdict"] == "MET"}
        v["mismatch"] = {arm: {k: mean([x.get(k) for x in R[arm]]) for k in ("mismatch_first10",)} for arm in ("S8G", "S4F")}
    if len(L4["S8LV"]) == 3 and len(L4["S8L"]) == 3:
        t8, tv = mean([x["t_gen_mean"] for x in L4["S8L"]]), mean([x["t_gen_mean"] for x in L4["S8LV"]])
        v["iv"] = {"S8L": t8, "S8LV": tv, "ratio": t8 / tv, "met": t8 <= 0.9 * tv}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "runs_q4l": L4, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
