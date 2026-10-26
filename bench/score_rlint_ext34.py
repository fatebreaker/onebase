"""RL-INTEGRATED extension -EXT34 (docs/REGISTRY.md, registered 2026-10-22 16:14 UTC before any data of RL-INTEGRATED): RL-INTEGRATED's
predictions with bench/score_rlint.py's arithmetic over every S8MS9L seed trained and evaluated by 2026-10-25 12:00 UTC (seeds 0-4),
against S8ML's seeds 0-2, reported beside the registered three-seed verdict, which stands.
  (i)   every S8MS9L run completes its 200 steps
  (ii)  S8MS9L's mean GPU-hours per run at most 0.90x S8ML's
  (iii) mean final accuracy (OlympiadBench+Minerva pooled, each on its own matched engine), S8MS9L minus S8ML: Welch 95% interval
        within +-5 points (else inconclusive)
  (iv)  S8MS9L's mean mismatch over the last 20 steps at most 1.25x S8ML's
    python3 bench/score_rlint_ext34.py [--json results/std_bench/rlint_ext34_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlint as R0  # noqa: E402  (its EV settings, deadline and helpers)
from score_rlint import DEADLINE, Q, evaluated, extra, welch  # noqa: E402

SEEDS = {"S8ML": (0, 1, 2), "S8MS9L": (0, 1, 2, 3, 4)}


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
            t = evaluated(arm, s)
            if x is None or x.get("hard_final") is None or (arm == "S8MS9L" and not (t and t <= DEADLINE)):
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            x.update(extra(arm, s), seed=s, evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None)
            R[arm].append(x)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    f, b = R["S8MS9L"], R["S8ML"]
    v: dict = {"steps": steps, "n": {arm: len(x) for arm, x in R.items()}}
    if len(f) >= 3 and len(b) == 3:
        v["i"] = {"met": all(steps[f"S8MS9L_s{x['seed']}"] >= 200 for x in f)}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["ii"] = {"S8MS9L": h1, "S8ML": h0, "ratio": h1 / h0, "met": h1 <= 0.9 * h0}
        v["iii"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
        m1, m0 = st.fmean(x["mismatch_last20"] for x in f), st.fmean(x["mismatch_last20"] for x in b)
        v["iv"] = {"S8MS9L": m1, "S8ML": m0, "ratio": m1 / m0, "met": m1 <= 1.25 * m0}
        v["at_or_below_pre_rl"] = sum(1 for x in f if x.get("hard_base") is not None and not x["hard_final"] > x["hard_base"])
    else:
        v["status"] = "incomplete"
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
