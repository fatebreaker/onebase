"""RL-H100-POWER (docs/REGISTRY.md, registered before any data): seeds 3-5 of the four H100 arms that carry the paper's RL accuracy
claims (S0TF the tuned bf16 baseline; S8TF the W8A8 engine with the plugin and the weight-matched trainer, TIS 2; S4TF FlashRL's
recipe, a bf16 trainer on the W8A8 engine, TIS 2; S8MH the matched W8A16 engine without correction), each run with its arm's
registered recipe and runner unchanged; analysed over every seed (0-5) trained and evaluated by 2026-10-24 00:00 UTC, at least
five per arm. Final accuracy: OlympiadBench+Minerva pooled, each arm on its own engine (bench/score_rlqmath.summary).
  (i)   every new run completes its 200 steps
  (ii)  S8TF minus S0TF: Welch 95% interval above -3 points (non-inferior at 3 points), and (ii-e) within +-3 (equivalent)
  (iii) S8MH minus S0TF: the same, (iii) and (iii-e)
Descriptive: S4TF minus S0TF, every arm's interval at three seeds against all seeds, GPU-hours per run.
    python3 bench/score_rlpower.py [--json results/std_bench/rlpower_verdict.json]"""
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
STOP = dt.datetime(2026, 10, 24, 0, 0, tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs = {arm: {k: Q.summary(arm, k) for k in range(6)} for arm in ARMS}
    done = {arm: {k: x for k, x in rs.items() if x and x.get("hard_final") is not None} for arm, rs in runs.items()}
    out = {"seeds": {arm: sorted(d) for arm, d in done.items()},
           "gpu_hours": {arm: [round(x["gpu_hours"], 3) for x in d.values()] for arm, d in done.items()}}
    new = {arm: [k for k in (3, 4, 5) if runs[arm][k] is not None] for arm in ARMS}
    final = all(len(d) >= 6 for d in done.values()) or (dt.datetime.now(dt.timezone.utc) >= STOP and all(len(d) >= 5 for d in done.values()))
    ref = [x["hard_final"] for x in done["S0TF"].values()]
    v: dict = {"final": final}
    v["i"] = {"met": all(len(new[arm]) == 3 for arm in ARMS), "new_runs_with_200_steps": new}
    for key, arm in (("ii", "S8TF"), ("iii", "S8MH"), ("descriptive_S4TF", "S4TF")):
        xs = [x["hard_final"] for x in done[arm].values()]
        w = welch(xs, ref) if len(xs) >= 2 and len(ref) >= 2 else None
        if w:
            w = {**w, "n": [len(xs), len(ref)]}
            if key != "descriptive_S4TF":
                w["met"] = -3.0 <= w["ci95"][0]                        # non-inferior at 3 points
                w["equivalent_met"] = -3.0 <= w["ci95"][0] and w["ci95"][1] <= 3.0
        three = welch([done[arm][k]["hard_final"] for k in (0, 1, 2) if k in done[arm]],
                      [done["S0TF"][k]["hard_final"] for k in (0, 1, 2) if k in done["S0TF"]])
        v[key] = {"all_seeds": w, "three_seeds": three}
    out["verdict"] = v
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
