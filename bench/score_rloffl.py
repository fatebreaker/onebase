"""RL-L40S-OFFLOAD (docs/REGISTRY.md, registered before any data): one-GPU LoRA RL of Qwen3.5-9B on a 48 GB L40S with sleep mode
and the trainer's base offloaded during rollouts on BOTH arms, at engine share 0.80 (train/grpo_q10.py, RL-L40S-FAST's recipe):
arm S8OL, the W8A8 engine with our plugin and the weight-matched trainer through our W8A16 kernels, TIS C = 2; arm S0OL, bf16
engine and trainer. Seeds 0-2 each.
  (i)   every run completes its 200 steps
  (ii)  mean GPU-hours per run, S8OL / S0OL, at most 0.90
  (iii) every run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled (own engine)
  (iv)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy (own engines), S8OL minus S0OL, within +-5 points
        (else inconclusive)
Stopping rule: the seeds with BOTH arms trained and evaluated by 2026-10-22 00:00 UTC (at least two; otherwise incomplete).
Descriptive: generation and trainer time per step, wake/sleep and load/offload, mismatch (first 10, last 20), TIS weights,
MATH-500 and AIME/AMC, against S8TLF (the same FP8 arithmetic without offload at 0.45).
Logs <adapters>/qwen9_dapo_<arm>_s<k>/log.jsonl; evaluations results/std_bench/qwen9_dapo_<arm>_s<k>/<variant>/<bench>.jsonl.
    python3 bench/score_rloffl.py [--json results/std_bench/rloffl_verdict.json]"""
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

DEADLINE = dt.datetime(2026, 10, 22, 0, 0, tzinfo=dt.timezone.utc)
Q.EV["S8OL"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0OL"] = ("bf16_base", "bf16_exact")
Q.EV["S8TLF"] = ("fp8eng_base", "fp8eng_exact")
BENCHES = ("olympiad", "minerva")


def evaluated_utc(arm: str, s: int) -> dt.datetime | None:
    """The latest modification time of the run's final-policy OlympiadBench/Minerva files (None if any is missing)."""
    t = None
    for b in BENCHES:
        f = Q.RES / f"qwen9_dapo_{arm}_s{s}/{Q.EV[arm][1]}/{b}.jsonl"
        if not f.exists():
            return None
        m = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)
        t = m if t is None or m > t else t
    return t


def extra(arm: str, s: int) -> dict:
    rs = Q.rows(arm, s)
    tr = [r["t"] - r["t_gen"] for r in rs]
    mm = [r.get("mismatch") for r in rs]
    tis = [r.get("tis_mean") for r in rs if r.get("tis_mean") is not None]
    keys = ("t_wake", "t_sleep", "t_load", "t_offload")
    return {"trainer_s": st.fmean(tr) if tr else None,
            "mismatch_last20": st.fmean([m for m in mm[-20:] if m is not None]) if any(m is not None for m in mm[-20:]) else None,
            "tis_mean": st.fmean(tis) if tis else None,
            **{f"{k}_mean": st.fmean(r.get(k, 0.0) for r in rs) for k in keys if rs and any(k in r for r in rs)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R, steps = {"S8OL": {}, "S0OL": {}}, {}
    for arm in R:
        for s in (0, 1, 2):
            rs = Q.rows(arm, s)
            steps[f"{arm}_s{s}"] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            if x is None or x.get("hard_final") is None:
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            t = evaluated_utc(arm, s)
            x.update(extra(arm, s), evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None, in_rule=bool(t and t <= DEADLINE))
            R[arm][s] = x
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    ref = [x for x in (Q.summary("S8TLF", s) for s in (0, 1, 2)) if x]
    pairs = [s for s in (0, 1, 2) if s in R["S8OL"] and s in R["S0OL"] and R["S8OL"][s]["in_rule"] and R["S0OL"][s]["in_rule"]]
    v = {"steps": steps, "seeds_used": pairs}
    if len(pairs) < 2:
        v["status"] = "incomplete (fewer than two seeds with both arms trained and evaluated by the stopping date)"
    else:
        f, b = [R["S8OL"][s] for s in pairs], [R["S0OL"][s] for s in pairs]
        v["i"] = {"met": all(steps[f"{arm}_s{s}"] >= 200 for arm in R for s in pairs)}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["ii"] = {"S8OL": h1, "S0OL": h0, "ratio": h1 / h0, "met": h1 <= 0.90 * h0}
        below = [f"{x['arm']}_s{x['seed']}" for x in f + b if not (x["hard_final"] > x["hard_base"])]
        v["iii"] = {"below": below, "met": not below}
        v["iv"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
        v["descriptive"] = {"t_gen": {"S8OL": st.fmean(x["t_gen_mean"] for x in f), "S0OL": st.fmean(x["t_gen_mean"] for x in b)},
                            "trainer_s": {"S8OL": st.fmean(x["trainer_s"] for x in f), "S0OL": st.fmean(x["trainer_s"] for x in b)},
                            "S8TLF_gpu_hours": st.fmean(x["gpu_hours"] for x in ref) if ref else None}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": {k: list(d.values()) for k, d in R.items()}, "S8TLF": ref, "verdict": v},
                                           indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
