"""RL-H100-POWER-2 (docs/REGISTRY.md, registered before any data): seeds 6-9 of the four H100 arms (S0TF, S8TF, S4TF, S8MH),
each run with RL-H100-POWER's runners unchanged; analysed over every seed (0-9) trained and evaluated by 2026-10-24 00:00 UTC, at
least eight per arm. Final accuracy: OlympiadBench+Minerva pooled, each arm on its own engine (bench/score_rlqmath.summary).
  (i)   every new run completes its 200 steps
  (ii)  S8TF minus S0TF: Welch 95% interval above -2.5 points (non-inferior at 2.5), and (ii-e) within +-2.5 (equivalent)
  (iii) S8MH minus S0TF: the same, (iii) and (iii-e)
Descriptive: S4TF minus S0TF; every arm's interval at three, six and all seeds; GPU-hours per run.
    python3 bench/score_rlpower2.py [--json results/std_bench/rlpower2_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

for _a in ("S8TF", "S4TF", "S8MH"):
    Q.EV[_a] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")
ARMS = ("S0TF", "S8TF", "S4TF", "S8MH")
SEEDS = range(10)
NEW = (6, 7, 8, 9)
MARGIN = 2.5
STOP = dt.datetime(2026, 10, 24, 0, 0, tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs = {arm: {k: Q.summary(arm, k) for k in SEEDS} for arm in ARMS}
    done = {arm: {k: x for k, x in rs.items() if x and x.get("hard_final") is not None} for arm, rs in runs.items()}
    out = {"seeds": {arm: sorted(d) for arm, d in done.items()},
           "gpu_hours": {arm: [round(x["gpu_hours"], 3) for x in d.values()] for arm, d in done.items()}}
    new = {arm: [k for k in NEW if runs[arm][k] is not None] for arm in ARMS}
    final = all(len(d) >= 10 for d in done.values()) or (dt.datetime.now(dt.timezone.utc) >= STOP and all(len(d) >= 8 for d in done.values()))
    ref = [x["hard_final"] for x in done["S0TF"].values()]
    v: dict = {"final": final}
    v["i"] = {"met": all(len(new[arm]) == len(NEW) for arm in ARMS), "new_runs_with_200_steps": new}
    for key, arm in (("ii", "S8TF"), ("iii", "S8MH"), ("descriptive_S4TF", "S4TF")):
        xs = [x["hard_final"] for x in done[arm].values()]
        w = welch(xs, ref) if len(xs) >= 2 and len(ref) >= 2 else None
        if w:
            w = {**w, "n": [len(xs), len(ref)]}
            if key != "descriptive_S4TF":
                w["met"] = -MARGIN <= w["ci95"][0]                     # non-inferior at 2.5 points
                w["equivalent_met"] = -MARGIN <= w["ci95"][0] and w["ci95"][1] <= MARGIN
        sub = {}
        for lab, ks in (("three_seeds", (0, 1, 2)), ("six_seeds", tuple(range(6)))):
            sub[lab] = welch([done[arm][k]["hard_final"] for k in ks if k in done[arm]],
                             [done["S0TF"][k]["hard_final"] for k in ks if k in done["S0TF"]])
        v[key] = {"all_seeds": w, **sub}
    out["verdict"] = v
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
