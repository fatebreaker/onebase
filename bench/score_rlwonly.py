"""RL-WONLY-A100 (docs/REGISTRY.md, registered 2026-10-18 14:30 UTC, before any data): arm S8WA, RL-FP8-QMATH's recipe on one
A100-80GB with vLLM's online FP8 engine (weight-only FP8 Marlin there), the weight-matched trainer with flash-linear-attention,
no importance correction (train/grpo_q4.py --engine fp8 --match fp8w), seeds 0-3. Logs <adapters>/qwen9_dapo_S8WA_s<k>/log.jsonl;
evaluations results/std_bench/qwen9_dapo_S8WA_s<k>/fp8eng_{base,exact} (bench/score_rlqmath.py's readers).
  (i)   every run completes its 200 steps
  (ii)  every run's mean trainer-engine mismatch over steps 181-200 is at most 0.010
  (iii) every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (iv)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy, S8WA - S0G (five H100 seeds), within +-5 (else
        inconclusive)
Stopping rule: the seeds whose training and evaluation finished by 2026-10-21 12:00 UTC (at least three), listed with their
evaluation times. Descriptive: mismatch over steps 1-10 and its maximum, generation and trainer time per step, GPU-hours,
MATH-500 and AIME24/25+AMC23 avg@16, S8WA against S8G per seed.
    python3 bench/score_rlwonly.py [--json results/std_bench/rlwonly_verdict.json]"""
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
import score_rltis as T  # noqa: E402
from score_rlmulti import welch  # noqa: E402

Q.EV["S8WA"] = ("fp8eng_base", "fp8eng_exact")
DEADLINE = dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, steps = [], {}
    for s in a.seeds:
        rs = Q.rows("S8WA", s)
        steps[s] = rs[-1]["step"] if rs else 0
        x = Q.summary("S8WA", s)
        et = T.eval_time("S8WA", s)
        if x is None or x.get("hard_final") is None or et is None:
            print(f"S8WA seed {s}: incomplete (steps {steps[s]})")
            continue
        x.update(T.extra("S8WA", s), eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE,
                 trainer_s=st.fmean(r["t"] - r["t_gen"] for r in rs))
        runs.append(x)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    s0 = [x for x in (Q.summary("S0G", k) for k in range(5)) if x and x.get("hard_final") is not None]
    s8 = {k: x for k, x in ((k, Q.summary("S8G", k)) for k in range(5)) if x and x.get("hard_final") is not None}
    use = [x for x in runs if x["in_rule"]]
    v = {"steps": steps, "seeds_used": [x["seed"] for x in use]}
    if len(use) < 3:
        v["status"] = "incomplete (fewer than three S8WA seeds by the stopping date)"
    else:
        v["i"] = {"met": all(steps[x["seed"]] >= 200 for x in use)}
        late = {x["seed"]: x["mismatch_last20"] for x in use}
        v["ii"] = {"mismatch_last20": late, "met": all(m is not None and m <= 0.010 for m in late.values())}
        below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
        v["iii"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                    "met": not below}
        v["iv"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in s0]) if len(s0) == 5 else \
            {"status": f"S0G has {len(s0)} of 5 seeds"}
        v["S8WA_vs_S8G_per_seed"] = {x["seed"]: {"S8WA": x["hard_final"], "S8G": s8[x["seed"]]["hard_final"]}
                                     for x in use if x["seed"] in s8}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8WA": runs, "S0G": s0, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
