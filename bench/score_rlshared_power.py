"""RL-SHARED-H100-POWER (docs/REGISTRY.md; this scorer written before any data of seeds 3-4): RL-SHARED (b)'s arm S8S27H over seeds
0-4, with bench/score_rlshared.py's helpers and reference (Q27, three benchmarks on the bf16 engine) unchanged.
  (i)  seeds 3 and 4 complete their 100 steps
  (ii) S8S27H (every seed trained and evaluated by the stopping date) minus Q27: Welch 95% interval within +-5 points
Descriptive: each seed's final accuracy and GPU-hours; the mean GPU-hours against S8T27H's.
    python3 bench/score_rlshared_power.py [--json results/std_bench/rlshared_power_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rl27 as R27  # noqa: E402
from score_rlmulti import welch  # noqa: E402
from score_rlshared import DEADLINE, evaluated3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    sh = []
    for k in range(5):
        x = R27.summary(f"qwen27_dapo_S8S27H_s{k}", 1)
        t = evaluated3(x["name"])
        x.update(seed=k, evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None, in_rule=bool(t and t <= DEADLINE))
        sh.append(x)
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()}))
    q = [R27.summary(f"qwen27_dapo_s{k}", 2)["final_bf16"] for k in (0, 1, 2)]
    v: dict = {"steps": [x["steps"] for x in sh]}
    new = [x for x in sh if x["seed"] >= 3]
    v["i"] = ("incomplete" if any(not x["steps"] for x in new) else
              {"met": all(x["steps"] >= 100 for x in new), "steps": [x["steps"] for x in new]})
    u = [x["final_bf16"] for x in sh if x["in_rule"] and x["final_bf16"] is not None]
    v["n_evaluated"] = len(u)
    v["ii"] = welch(u, [y for y in q if y is not None]) if len(u) >= 4 else "incomplete"
    gh = [x["gpu_hours"] for x in sh if x["steps"] >= 100 and x["gpu_hours"]]
    v["gpu_hours_mean"] = st.fmean(gh) if gh else None
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8S27H": sh, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
