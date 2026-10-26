"""RL-L40S-OFFLOAD's extensions (docs/REGISTRY.md: extension, registered 2026-10-21 06:05 UTC, seeds 3-4; extension 2, 13:22 UTC,
seeds 5-6; both descriptive): (iv)'s interval, S8OL minus S0OL (Welch 95%, mean final OlympiadBench+Minerva on each arm's own
engine), over every seed trained and evaluated by 2026-10-22 18:00 UTC, labelled as an extension; seeds not finished by then are
listed as incomplete. The registered verdict (seeds 0-2, bench/score_rloffl.py) stands.
    python3 bench/score_rloffl_ext.py [--json results/std_bench/rloffl_ext_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402
from score_rloffl import evaluated_utc  # noqa: E402  (also sets Q.EV for S8OL and S0OL)

DEADLINE = dt.datetime(2026, 10, 22, 18, 0, tzinfo=dt.timezone.utc)
SEEDS = range(7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, incomplete = {"S8OL": {}, "S0OL": {}}, []
    for arm in runs:
        for s in SEEDS:
            x = Q.summary(arm, s)
            t = evaluated_utc(arm, s)
            if x is None or x.get("hard_final") is None or not (t and t <= DEADLINE):
                incomplete.append(f"{arm} seed {s}")
                continue
            runs[arm][s] = {"final": x["hard_final"], "gpu_hours": x["gpu_hours"], "evaluated_utc": t.strftime("%Y-%m-%dT%H:%M")}
    f, b = list(runs["S8OL"].values()), list(runs["S0OL"].values())
    v = {"seeds": {arm: sorted(d) for arm, d in runs.items()}, "incomplete": incomplete}
    if len(f) >= 2 and len(b) >= 2:
        v["iv_ext"] = welch([x["final"] for x in f], [x["final"] for x in b])
        v["gpu_hours_ratio"] = st.fmean(x["gpu_hours"] for x in f) / st.fmean(x["gpu_hours"] for x in b)
    print(json.dumps({"runs": runs, "verdict": v}, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": runs, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
