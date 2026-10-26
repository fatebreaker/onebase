"""RL-MULTI-MATH (docs/REGISTRY.md): three pipelined GRPO jobs on one FP8 engine (train/grpo_multi.py; each job's log in
<adapters>/llama_math_S9M/job<k>/log.jsonl, its MATH-500 on the FP8 engine in results/std_bench/llama_math_S9M_s<k>/
fp8eng_exact) against RL-FP8-MATH's single-job runs (bench/score_rlmath.py's run()).
  (i)   GPU-hours per run (a job's summed "t" over its 200 steps) at most 0.8x the mean of S8's runs
  (ii)  each job's mean trainer-engine |delta log p| over steps 1-10 below 0.02
  (iii) Welch 95% interval of mean final MATH-500 accuracy, these jobs minus S0's runs, within +-5 points
--arm S9B: the batched amendment (llama_math_S9B), judged by the same rules.
    python3 bench/score_rlmulti.py [--arm S9M|S9B] [--json results/std_bench/rlmulti_verdict.json]"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
import score_rlmath as sm  # noqa: E402

ADS = Path("$SCRATCH/std_bench/adapters")


def welch(x: list[float], y: list[float]) -> dict:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    vx = sum((t - mx) ** 2 for t in x) / (len(x) - 1)
    vy = sum((t - my) ** 2 for t in y) / (len(y) - 1)
    se = math.sqrt(vx / len(x) + vy / len(y))
    df = se ** 4 / ((vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1)) if se > 0 else 1e9
    q = sm.t_quantile(0.975, df) if se > 0 else 0.0
    lo, hi = mx - my - q * se, mx - my + q * se
    return {"diff": mx - my, "ci95": [lo, hi], "df": df,
            "verdict": "MET" if -5 <= lo and hi <= 5 else ("INCONCLUSIVE" if lo < 5 and hi > -5 else "NOT MET")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    ap.add_argument("--arm", default="S9M", choices=("S9M", "S9B"))
    a = ap.parse_args()
    AD = ADS / f"llama_math_{a.arm}"
    jobs = []
    for k, s in enumerate(a.seeds):
        f = AD / f"job{k}/log.jsonl"
        rows = runlog.read(f)                                  # one row per step (bench/runlog.py)
        if not rows or rows[-1]["step"] < 200:
            print(f"job {k} (seed {s}): not complete ({rows[-1]['step'] if rows else 0} steps)")
            continue
        mm = [r["mismatch"] for r in rows if r["step"] <= 10 and r.get("mismatch") is not None]
        jobs.append({"seed": s, "gpu_hours": sum(r["t"] for r in rows) / 3600, "mismatch_first10": sum(mm) / len(mm),
                     "t_gen_mean": sum(r["t_gen"] for r in rows) / len(rows), "t_train_mean": sum(r["t_train"] for r in rows) / len(rows),
                     "reward_early": sum(r["reward"] for r in rows if r["step"] <= 50) / 50,
                     "reward_late": sum(r["reward"] for r in rows if r["step"] > 150) / 50,
                     "acc_final": sm.acc(sm.RES / f"llama_math_{a.arm}_s{s}/fp8eng_exact/math500.jsonl"),
                     "wall_hours": rows[-1]["wall"] / 3600})
        print(json.dumps(jobs[-1]))
    s0 = [r for r in (sm.run("S0", s) for s in a.seeds) if r]
    s8 = [r for r in (sm.run("S8", s) for s in a.seeds) if r]
    v = {}
    if len(jobs) == len(a.seeds) and s8:
        m8 = sum(r["gpu_hours"] for r in s8) / len(s8)
        mj = sum(j["gpu_hours"] for j in jobs) / len(jobs)
        v["i"] = {"multi": mj, "S8": m8, "ratio": mj / m8, "verdict": "MET" if mj <= 0.8 * m8 else "NOT MET"}
        v["ii"] = {"per_job": [j["mismatch_first10"] for j in jobs],
                   "verdict": "MET" if all(j["mismatch_first10"] < 0.02 for j in jobs) else "NOT MET"}
        x = [j["acc_final"] for j in jobs if j["acc_final"] is not None]
        y = [r["acc_final"] for r in s0 if r["acc_final"] is not None]
        if len(x) >= 2 and len(y) >= 2:
            v["iii"] = welch(x, y)
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"jobs": jobs, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
