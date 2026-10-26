"""RL-27B-H100-PAIR (docs/REGISTRY.md; this scorer written before any data of seeds 3-4 of S8T27H): one copy (S8S27H, RL-SHARED (b)
and its power extension) minus two copies (S8T27H, RL-27B-H100), Qwen3.8-27B on one H100, paired by seed over the seeds where both
completed 100 steps and were evaluated by 2026-10-25 12:00 UTC.
  (i) the mean paired difference of final accuracy (MATH-500, OlympiadBench and Minerva pooled on the bf16 engine), paired 95%
      t-interval within +-5 points (else inconclusive)
Descriptive: each pair; the paired GPU-hours ratio; two-copy runs that ran out of memory.
    python3 bench/score_rlpair.py [--json results/std_bench/rlpair_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rl27 as R27  # noqa: E402
import score_rlmath as sm  # noqa: E402  (t_quantile)

DEADLINE = dt.datetime(2026, 10, 25, 12, 0, tzinfo=dt.timezone.utc)
SEEDS = (0, 1, 2, 3, 4, 5, 6)   # seeds 5-6: extension -EXT56 (registered 2026-10-22 16:10 UTC: every pair completed counts)


def evaluated(name: str) -> bool:
    fs = [LB / f"results/std_bench/{name}/bf16_exact/{b}.jsonl" for b in ("math500", "olympiad", "minerva")]
    return all(f.exists() for f in fs) and max(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc) for f in fs) <= DEADLINE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    pairs, oom = [], []
    for k in SEEDS:
        one, two = R27.summary(f"qwen27_dapo_S8S27H_s{k}", 1), R27.summary(f"qwen27_dapo_S8T27H_s{k}", 1)
        if two["steps"] and two["steps"] < 100:
            oom.append(k)
        ok = all(x["steps"] >= 100 and x["final_bf16"] is not None and evaluated(x["name"]) for x in (one, two))
        if ok:
            pairs.append({"seed": k, "one": one["final_bf16"], "two": two["final_bf16"], "diff": one["final_bf16"] - two["final_bf16"],
                          "gh_one": one["gpu_hours"], "gh_two": two["gpu_hours"]})
        print(json.dumps({"seed": k, "one_steps": one["steps"], "two_steps": two["steps"], "paired": ok}))
    v: dict = {"pairs": pairs, "two_copy_short_runs": oom}
    if len(pairs) >= 3:
        d = [p["diff"] for p in pairs]
        m, sd = st.fmean(d), st.stdev(d)
        q = sm.t_quantile(0.975, len(d) - 1)
        lo, hi = m - q * sd / math.sqrt(len(d)), m + q * sd / math.sqrt(len(d))
        v["i"] = {"n_pairs": len(d), "diff": m, "ci95": [lo, hi],
                  "verdict": "MET" if -5 <= lo and hi <= 5 else ("INCONCLUSIVE" if lo < 5 and hi > -5 else "NOT MET")}
        v["gpu_hours_ratio"] = st.fmean(p["gh_one"] / p["gh_two"] for p in pairs)
    else:
        v["status"] = f"{len(pairs)} pairs"
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
