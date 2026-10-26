"""RL-L40S (docs/REGISTRY.md, registered 2026-10-17): arm S8TL (S8T's recipe on one 48 GB L40S), seeds 0-2.
  (i)   every run completes its 200 steps
  (ii)  every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (iii) Welch 95% interval of mean final accuracy, S8TL - S0G (five seeds, H100), within +-5 (else inconclusive)
Descriptive: step and generation times, GPU-hours, mismatch, TIS weight statistics, against S8G and S0T.
    python3 bench/score_rll40.py [--json results/std_bench/rll40_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
import score_rltis as T  # noqa: E402
from score_rlmulti import welch  # noqa: E402

Q.EV["S8TL"] = ("fp8eng_base", "fp8eng_exact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, steps = [], {}
    for s in (0, 1, 2):
        rows = Q.rows("S8TL", s)
        steps[s] = rows[-1]["step"] if rows else 0
        x = Q.summary("S8TL", s)
        if x is None or x.get("hard_final") is None:
            print(f"S8TL seed {s}: incomplete (steps {steps[s]})")
            continue
        x.update(T.extra("S8TL", s))
        runs.append(x)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    s0 = [x for x in (Q.summary("S0G", k) for k in range(5)) if x and x.get("hard_final") is not None]
    v = {"steps": steps}
    if len(runs) == 3:
        v["i"] = {"met": all(steps[s] >= 200 for s in (0, 1, 2))}
        below = [x["seed"] for x in runs if not x["hard_final"] > x["hard_base"]]
        v["ii"] = {"below": below, "met": not below}
        v["iii"] = welch([x["hard_final"] for x in runs], [x["hard_final"] for x in s0]) if len(s0) == 5 else {"status": "S0G incomplete"}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8TL": runs, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
