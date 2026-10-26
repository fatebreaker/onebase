"""RL-MATCHED (docs/REGISTRY.md, registered 2026-10-18 19:26 UTC, before any data): arm S8ML, RL-L40S-FAST's S8TLF recipe on one
48 GB L40S with the matched-precision engine (train/grpo_q6.py --w16-engine: vLLM's online FP8 layers W8A16 through our kernel,
in-process) and no importance correction, top-32 diagnostics, seeds 0-2. Logs <adapters>/qwen9_dapo_S8ML_s<k>/log.jsonl;
evaluations results/std_bench/qwen9_dapo_S8ML_s<k>/fp8eng_{base,exact} (on the same engine, bench/std_bench_w16.py).
  (i)   every run completes its 200 steps
  (ii)  every run's mean trainer-engine mismatch over steps 181-200 is at most 0.010
  (iii) every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (iv)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy, S8ML - S0G (five H100 seeds), within +-5 (else
        inconclusive)
Stopping rule: the seeds whose training and evaluation finished by 2026-10-22 00:00 UTC (at least two). Descriptive: tv, kl,
head per step; generation and trainer time per step and GPU-hours against S8TLF; S8ML against S8TLF per seed.
    python3 bench/score_rlmatched.py [--json results/std_bench/rlmatched_verdict.json]"""
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

Q.EV["S8ML"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S8TLF"] = ("fp8eng_base", "fp8eng_exact")
DEADLINE = dt.datetime(2026, 10, 22, 0, 0, tzinfo=dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, steps = [], {}
    for s in a.seeds:
        rs = Q.rows("S8ML", s)
        steps[s] = rs[-1]["step"] if rs else 0
        x = Q.summary("S8ML", s)
        et = T.eval_time("S8ML", s)
        if x is None or x.get("hard_final") is None or et is None:
            print(f"S8ML seed {s}: incomplete (steps {steps[s]})")
            continue
        late = lambda k: st.fmean([r[k] for r in rs if r["step"] > 180 and r.get(k) is not None] or [float("nan")])  # noqa: E731
        early = lambda k: st.fmean([r[k] for r in rs if r["step"] <= 10 and r.get(k) is not None] or [float("nan")])  # noqa: E731
        x.update(T.extra("S8ML", s), eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE,
                 trainer_s=st.fmean(r["t"] - r["t_gen"] for r in rs), tv_first10=early("tv"), tv_last20=late("tv"),
                 kl_first10=early("kl"), kl_last20=late("kl"))
        runs.append(x)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    s0 = [x for x in (Q.summary("S0G", k) for k in range(5)) if x and x.get("hard_final") is not None]
    s8 = {k: x for k, x in ((k, Q.summary("S8TLF", k)) for k in range(3)) if x and x.get("hard_final") is not None}
    use = [x for x in runs if x["in_rule"]]
    v = {"steps": steps, "seeds_used": [x["seed"] for x in use]}
    if len(use) < 2:
        v["status"] = "incomplete (fewer than two S8ML seeds by the stopping date)"
    else:
        v["i"] = {"met": all(steps[x["seed"]] >= 200 for x in use)}
        late = {x["seed"]: x["mismatch_last20"] for x in use}
        v["ii"] = {"mismatch_last20": late, "met": all(m is not None and m <= 0.010 for m in late.values())}
        below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
        v["iii"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                    "met": not below}
        v["iv"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in s0]) if len(s0) == 5 else \
            {"status": f"S0G has {len(s0)} of 5 seeds"}
        v["S8ML_vs_S8TLF_per_seed"] = {x["seed"]: {"S8ML": x["hard_final"], "S8TLF": s8[x["seed"]]["hard_final"]}
                                     for x in use if x["seed"] in s8}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8ML": runs, "S0G": s0, "S8TLF": list(s8.values()), "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
