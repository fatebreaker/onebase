"""RL-L40S-OFFLOAD-S4 (docs/REGISTRY.md, registered before any data): does the weight-matched FP8 trainer contribute to
RL-L40S-OFFLOAD's GPU-hour saving? Arm S4OL: the same offload recipe (train/grpo_q10.py, sleep mode, the trainer's base offloaded
during rollouts, engine share 0.80) with vLLM's FP8 engine and our plugin, a bf16 trainer without weight matching, TIS C = 2;
seeds 0-2. References: RL-L40S-OFFLOAD's S8OL (FP8 engine, weight-matched FP8 trainer, TIS) and S0OL (bf16 both sides), every
seed trained and evaluated by the stopping date.
  (i)   every S4OL run completes its 200 steps
  (ii)  mean GPU-hours per run, S4OL / S0OL, at most 0.90 (the FP8 engine alone carries the saving)
  (iii) mean GPU-hours per run, S4OL / S8OL, within 0.95-1.05 (the FP8 trainer adds no speed under offload)
  (iv)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy (own engines), S4OL minus S0OL, within +-5 points
        (else inconclusive)
Stopping rule: the S4OL seeds trained and evaluated by 2026-10-23 12:00 UTC (at least two; otherwise incomplete).
Descriptive: generation, trainer and swap time per step; mismatch over steps 1-10 and 181-200; TIS weights; S4OL minus S8OL
(Welch); MATH-500 and AIME/AMC; the fit probes (logs/lb_qoffl6_fit_u*.out).
    python3 bench/score_rloffl_s4.py [--json results/std_bench/rloffl_s4_verdict.json]"""
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
from score_rlmulti import welch  # noqa: E402
from score_rloffl import evaluated_utc, extra  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)
Q.EV["S4OL"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S8OL"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0OL"] = ("bf16_base", "bf16_exact")


def runs(arm: str, seeds) -> tuple[dict, dict]:
    out, steps = {}, {}
    for s in seeds:
        rs = Q.rows(arm, s)
        steps[s] = rs[-1]["step"] if rs else 0
        x = Q.summary(arm, s)
        if x is None or x.get("hard_final") is None:
            continue
        t = evaluated_utc(arm, s)
        if not (t and t <= DEADLINE):
            continue
        rs10 = [r.get("mismatch") for r in rs[:10] if r.get("mismatch") is not None]
        x.update(extra(arm, s), evaluated_utc=t.strftime("%Y-%m-%dT%H:%M"),
                 mismatch_first10=st.fmean(rs10) if rs10 else None)
        out[s] = x
    return out, steps


def fit_probes() -> dict:
    """Each probe's outcome from its log: completed (two steps logged) or the last memory error seen."""
    out = {}
    for f in sorted((LB / "logs").glob("lb_qoffl6_fit_u*.out")):
        t = f.read_text(errors="replace")
        u = f.stem.split("_u")[-1]
        done = t.count('"step": 2,') > 0
        err = [x for x in ("OutOfMemoryError", "out of memory", "No available memory") if x in t]
        out[u] = "completed 2 steps" if done else ("out of memory" if err else "did not complete")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    s4, steps = runs("S4OL", (0, 1, 2))
    s8, _ = runs("S8OL", range(10))
    s0, _ = runs("S0OL", range(10))
    for x in s4.values():
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {"steps": steps, "seeds_used": sorted(s4), "S8OL_seeds": sorted(s8), "S0OL_seeds": sorted(s0)}
    if len(s4) < 2:
        v["status"] = "incomplete (fewer than two S4OL seeds trained and evaluated by the stopping date)"
    else:
        f, e, b = list(s4.values()), list(s8.values()), list(s0.values())
        h4, h8, h0 = (st.fmean(x["gpu_hours"] for x in xs) for xs in (f, e, b))
        v["i"] = {"met": all(steps[s] >= 200 for s in s4)}
        v["ii"] = {"S4OL": h4, "S0OL": h0, "ratio": h4 / h0, "met": h4 <= 0.90 * h0}
        v["iii"] = {"S4OL": h4, "S8OL": h8, "ratio": h4 / h8, "met": 0.95 <= h4 / h8 <= 1.05}
        v["iv"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
        mean = lambda xs, k: st.fmean(x[k] for x in xs if x.get(k) is not None) if any(x.get(k) is not None for x in xs) else None  # noqa: E731
        v["descriptive"] = {
            "S4OL_minus_S8OL": welch([x["hard_final"] for x in f], [x["hard_final"] for x in e]),
            **{k: {arm: mean(xs, k) for arm, xs in (("S4OL", f), ("S8OL", e), ("S0OL", b))}
               for k in ("t_gen_mean", "trainer_s", "mismatch_first10", "mismatch_last20", "tis_mean", "t_load_mean",
                         "t_offload_mean", "t_wake_mean", "t_sleep_mean", "math500_final", "aime_amc_final")},
            "fit_probes": fit_probes()}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S4OL": list(s4.values()), "S8OL": list(s8.values()), "S0OL": list(s0.values()),
                                            "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
