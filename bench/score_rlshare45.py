"""RL-SHARE45 (docs/REGISTRY.md; this scorer written before any data): arm S8MS45 = RL-INTEGRATED's S8MS9L recipe (one shared FP8
copy, the matched engine in process, no correction; train/grpo_q15.py --shared-base) at engine share 0.45, the two-copy S8ML's,
instead of 0.55; seeds 0-2, against S8ML seeds 0-2 (two copies, the same share and arithmetic) and S8MS9L seeds 0-4 (one copy at 0.55).
  (i)  every S8MS45 run completes its 200 steps
  (ii) mean final accuracy (OlympiadBench+Minerva pooled, each policy on its own matched engine), S8MS45 minus S8ML: Welch 95%
       interval within +-5 points (else inconclusive)
Descriptive: S8MS45 minus S8MS9L (Welch); against S0OL (bf16, the same GPU); GPU-hours, generation and trainer time, the engine's
KV cache, the mismatch over the last 20 steps; the runs at or below their pre-RL accuracy. Runs trained and evaluated by
2026-10-26 12:00 UTC count.
    python3 bench/score_rlshare45.py [--json results/std_bench/rlshare45_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlint as I  # noqa: E402
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.timezone.utc)
SEEDS = {"S8MS45": (0, 1, 2), "S8ML": (0, 1, 2), "S8MS9L": (0, 1, 2, 3, 4)}
for arm in SEEDS:
    Q.EV[arm] = ("fp8eng_base", "fp8eng_exact")
Q.EV.setdefault("S0OL", ("fp8eng_base", "fp8eng_exact"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R, steps = {arm: [] for arm in SEEDS}, {}
    for arm, seeds in SEEDS.items():
        for s in seeds:
            rs = Q.rows(arm, s)
            steps[f"{arm}_s{s}"] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            t = I.evaluated(arm, s)
            if x is None or x.get("hard_final") is None or (arm == "S8MS45" and not (t and t <= DEADLINE)):
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            x.update(I.extra(arm, s))
            R[arm].append(x)
    v: dict = {"steps": {k: v for k, v in steps.items() if k.startswith("S8MS45")}}
    if len(R["S8MS45"]) == 3 and len(R["S8ML"]) == 3:
        f = R["S8MS45"]
        v["i"] = {"met": all(steps[f"S8MS45_s{s}"] >= 200 for s in SEEDS["S8MS45"])}
        v["ii"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in R["S8ML"]])
        s0 = [x["hard_final"] for x in (Q.summary("S0OL", k) for k in range(7)) if x and x.get("hard_final") is not None]
        v["descriptive"] = {
            "minus_S8MS9L": welch([x["hard_final"] for x in f], [x["hard_final"] for x in R["S8MS9L"]]) if len(R["S8MS9L"]) >= 2 else None,
            "minus_S0OL": welch([x["hard_final"] for x in f], s0) if len(s0) >= 2 else None,
            "gpu_hours": {arm: st.fmean(x["gpu_hours"] for x in R[arm]) for arm in R if R[arm]},
            "t_gen": {arm: st.fmean(x["t_gen_mean"] for x in R[arm]) for arm in R if R[arm]},
            "trainer_s": {arm: st.fmean(x["trainer_s"] for x in R[arm]) for arm in R if R[arm]},
            "mismatch_last20": {arm: st.fmean(x["mismatch_last20"] for x in R[arm]) for arm in R if R[arm]},
            "at_or_below_pre_rl": sum(1 for x in f if x.get("hard_base") is not None and not x["hard_final"] > x["hard_base"])}
    else:
        v["status"] = "incomplete"
    out = {"runs": {arm: [round(x["hard_final"], 2) for x in R[arm]] for arm in R}, "verdict": v}
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
