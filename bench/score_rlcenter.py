"""RL-CENTER (docs/REGISTRY.md, registered 2026-10-18 15:29 UTC, before any data): Qwen3.5-9B math RL on the lent H100 with
RL-H100-FAST's fast trainer; arms S8C (W8A8 FP8 engine with our plugin, score centering on the engine's top-32,
train/grpo_q5.py --center 32) and S8W (weight-only FP8 engine, --linear-backend marlin, no correction, --diag-topk 32), seeds
0-3. Logs <adapters>/qwen9_dapo_<arm>_s<k>/log.jsonl; evaluations results/std_bench/qwen9_dapo_<arm>_s<k>/fp8eng_{base,exact}.
Per arm:
  (i)   every run completes its 200 steps
  (ii)  every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (iii) Welch 95% interval of mean final OlympiadBench+Minerva accuracy, arm - S0G (five seeds), within +-5 (else inconclusive)
and (iv) S8W's mean tv and mismatch over steps 181-200 are each at most half of S8C's.
Stopping rule: the seeds whose training and evaluation finished by 2026-10-23 12:00 UTC (at least three per arm).
Descriptive: tv, kl, head and the centering term per step; GPU-hours against S8TF and S0TF (RL-H100-FAST); per seed against
S8G and S8T.
    python3 bench/score_rlcenter.py [--json results/std_bench/rlcenter_verdict.json]"""
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

ARMS = ("S8C", "S8W")
for _a in ARMS + ("S8TF", "S8T"):
    Q.EV[_a] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")
DEADLINE = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def late(rs: list[dict], key: str) -> float | None:
    v = [r[key] for r in rs if r["step"] > 180 and r.get(key) is not None]
    return st.fmean(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out, v = {}, {}
    s0 = [x for x in (Q.summary("S0G", k) for k in range(5)) if x and x.get("hard_final") is not None]
    for arm in ARMS:
        runs, steps = [], {}
        for s in (0, 1, 2, 3):
            rs = Q.rows(arm, s)
            steps[s] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            et = T.eval_time(arm, s)
            if x is None or x.get("hard_final") is None or et is None:
                print(f"{arm} seed {s}: incomplete (steps {steps[s]})")
                continue
            x.update(T.extra(arm, s), eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE,
                     tv_first10=st.fmean([r["tv"] for r in rs if r["step"] <= 10 and r.get("tv") is not None] or [float("nan")]),
                     tv_last20=late(rs, "tv"), kl_last20=late(rs, "kl"), center_l1_last20=late(rs, "center_l1"),
                     trainer_s=st.fmean(r["t"] - r["t_gen"] for r in rs))
            runs.append(x)
            print(json.dumps({k: (round(y, 4) if isinstance(y, float) else y) for k, y in x.items()}))
        use = [x for x in runs if x["in_rule"]]
        w = {"steps": steps, "seeds_used": [x["seed"] for x in use]}
        if len(use) < 3:
            w["status"] = f"incomplete (fewer than three {arm} seeds by the stopping date)"
        else:
            w["i"] = {"met": all(steps[x["seed"]] >= 200 for x in use)}
            below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
            w["ii"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                       "met": not below}
            w["iii"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in s0]) if len(s0) == 5 else \
                {"status": f"S0G has {len(s0)} of 5 seeds"}
        out[arm], v[arm] = use, w
    if all("i" in v[arm] for arm in ARMS):
        m = {arm: {k: st.fmean([x[k] for x in out[arm] if x.get(k) is not None]) for k in ("tv_last20", "mismatch_last20")}
             for arm in ARMS}
        v["iv"] = {"S8W": m["S8W"], "S8C": m["S8C"],
                   "met": all(m["S8W"][k] <= 0.5 * m["S8C"][k] for k in ("tv_last20", "mismatch_last20"))}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": out, "S0G": s0, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
