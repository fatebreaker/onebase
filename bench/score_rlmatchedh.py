"""RL-MATCHED-H100 (docs/REGISTRY.md, registered before any data): arm S8MH, RL-H100-FAST's S8TF recipe on one H100 with the
matched engine (train/grpo_q11.py --w16-engine, the swap-AB GEMM if MATCHED-HOPPER-KL adopted it) and NO importance correction,
seeds 0-2, against S0TF (RL-H100-FAST's tuned bf16 arm: the same GPU and trainer kernels, three seeds).
  (i)   every S8MH run completes its 200 steps
  (ii)  every run's mean trainer-engine mismatch over steps 181-200 is at most 0.010
  (iii) every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled (own engine)
  (iv)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy, S8MH minus S0TF, within +-5 points (else inconclusive)
Stopping rule: the seeds trained and evaluated by 2026-10-22 00:00 UTC (at least two). Descriptive: GPU-hours against S8TF
(W8A8 engine with our plugin, TIS) and S0TF; late mismatch against S8G (the W8A8 engine without correction, one collapse in five).
    python3 bench/score_rlmatchedh.py [--json results/std_bench/rlmatchedh_verdict.json]"""
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

Q.EV["S8MH"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S8TF"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")
DEADLINE = dt.datetime(2026, 10, 22, 0, 0, tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, steps = [], {}
    for s in a.seeds:
        rs = Q.rows("S8MH", s)
        steps[s] = rs[-1]["step"] if rs else 0
        x = Q.summary("S8MH", s)
        et = T.eval_time("S8MH", s)
        if x is None or x.get("hard_final") is None or et is None:
            print(f"S8MH seed {s}: incomplete (steps {steps[s]})")
            continue
        late = lambda k: st.fmean([r[k] for r in rs if r["step"] > 180 and r.get(k) is not None] or [float("nan")])  # noqa: E731
        early = lambda k: st.fmean([r[k] for r in rs if r["step"] <= 10 and r.get(k) is not None] or [float("nan")])  # noqa: E731
        x.update(T.extra("S8MH", s), eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE,
                 trainer_s=st.fmean(r["t"] - r["t_gen"] for r in rs), tv_first10=early("tv"), tv_last20=late("tv"),
                 kl_first10=early("kl"), kl_last20=late("kl"))
        runs.append(x)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    s0 = [x for x in (Q.summary("S0TF", k) for k in range(3)) if x and x.get("hard_final") is not None]
    s8 = {k: x for k, x in ((k, Q.summary("S8TF", k)) for k in range(3)) if x and x.get("hard_final") is not None}
    use = [x for x in runs if x["in_rule"]]
    v = {"steps": steps, "seeds_used": [x["seed"] for x in use]}
    if len(use) < 2:
        v["status"] = "incomplete (fewer than two S8MH seeds by the stopping date)"
    else:
        v["i"] = {"met": all(steps[x["seed"]] >= 200 for x in use)}
        late = {x["seed"]: x["mismatch_last20"] for x in use}
        v["ii"] = {"mismatch_last20": late, "met": all(m is not None and m <= 0.010 for m in late.values())}
        below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
        v["iii"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                    "met": not below}
        v["iv"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in s0]) if len(s0) == 3 else \
            {"status": f"S0TF has {len(s0)} of 3 seeds"}
        v["gpu_hours"] = {"S8MH": st.fmean(x["gpu_hours"] for x in use), "S8TF": st.fmean(x["gpu_hours"] for x in s8.values()) if s8 else None,
                          "S0TF": st.fmean(x["gpu_hours"] for x in s0) if s0 else None}
        v["late_mismatch_S8G"] = [T.extra("S8G", k)["mismatch_last20"] for k in range(5)]
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8MH": runs, "S0TF": s0, "S8TF": list(s8.values()), "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
