"""EVAL-AVG8-H100ALL (docs/REGISTRY.md, registered before any data of the policies it adds; written before any data): every final
9B policy of the paper's arms on the H100's bf16 engine with avg@8 (EVAL-BF16-AVG8's protocol: results/std_bench_avg8/<run>/
bf16_exact/{olympiad,minerva}.jsonl), with amendment -PAIRS.
  (i) engine equivalence: over the policies scored on both the GH200's engine (results/std_bench_avg8_gh200/) and the H100's,
      the paired mean difference H100 minus GH200 has a 95% interval inside +-1.5 points
Descriptive: each arm's mean and 95% interval (t, over its scored seeds) on the H100 engine. Policies scored by 2026-10-25 12:00
UTC count.
    python3 bench/score_avg8_h100all.py [--json results/std_bench/avg8_h100all_verdict.json]"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_avg8_onecopy as O  # noqa: E402

H100 = LB / "results/std_bench_avg8"
GH200 = LB / "results/std_bench_avg8_gh200"
ARMS = ("S0OL", "S8OL", "S4OL", "S8K9L", "S8ML", "S8MS9L", "S8S9L", "S8TLF", "S0TF", "S4TF", "S8MH", "S8TF", "S0G", "S8G", "S8C",
        "S8W", "S8RA", "S8ULF", "S0T", "S8T", "S8TL", "S4F", "S8WA", "S8UF")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        25: 2.060, 30: 2.042}


def t975(df: int) -> float:
    return T975.get(df) or min((v for k, v in T975.items() if k >= df), default=1.96)


def at(root: Path, arm: str, s: int) -> float | None:
    O.RES = root
    return O.score(arm, s)


def ci(xs: list) -> dict:
    m = st.fmean(xs)
    if len(xs) < 2:
        return {"n": len(xs), "mean": m}
    h = t975(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs))
    return {"n": len(xs), "mean": m, "ci95": [m - h, m + h]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    arms, diffs = {}, []
    for arm in ARMS:
        xs = {}
        for s in range(10):
            h = at(H100, arm, s)
            if h is None:
                continue
            xs[s] = h
            g = at(GH200, arm, s)
            if g is not None:
                diffs.append({"policy": f"{arm}_s{s}", "h100": h, "gh200": g, "diff": h - g})
        if xs:
            arms[arm] = {**ci(list(xs.values())), "seeds": sorted(xs)}
    v: dict = {"pairs": len(diffs)}
    if len(diffs) >= 2:
        d = ci([x["diff"] for x in diffs])
        v["i"] = {**d, "met": -1.5 <= d["ci95"][0] and d["ci95"][1] <= 1.5}
    out = {"arms": arms, "pairs": diffs, "verdict": v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
