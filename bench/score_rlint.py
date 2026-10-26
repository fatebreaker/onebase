"""RL-INTEGRATED (docs/REGISTRY.md; this scorer written before any data): arm S8MS9L = RL-MATCHED's S8ML recipe on one L40S with one
shared FP8 copy (train/grpo_q15.py --shared-base with the in-process matched engine, engine share 0.55), no correction, seeds 0-2,
against S8ML seeds 0-2 (two copies, the same arithmetic).
  (i)   every S8MS9L run completes its 200 steps
  (ii)  S8MS9L's mean GPU-hours per run at most 0.90x S8ML's
  (iii) mean final accuracy on OlympiadBench+Minerva pooled (each on its own matched engine), S8MS9L minus S8ML: Welch 95% interval
        within +-5 points (else inconclusive)
  (iv)  S8MS9L's mean mismatch over the last 20 steps at most 1.25x S8ML's
Runs trained and evaluated by 2026-10-25 12:00 UTC count. Descriptive: S8MS9L minus S0G (bf16 on the H100), generation and trainer
time, mismatch over steps 1-10, AIME/AMC, the runs at or below their pre-RL accuracy.
    python3 bench/score_rlint.py [--json results/std_bench/rlint_verdict.json]"""
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

DEADLINE = dt.datetime(2026, 10, 25, 12, 0, tzinfo=dt.timezone.utc)
for arm in ("S8ML", "S8MS9L"):
    Q.EV[arm] = ("fp8eng_base", "fp8eng_exact")


def evaluated(arm: str, s: int) -> dt.datetime | None:
    fs = [Q.RES / f"qwen9_dapo_{arm}_s{s}/fp8eng_exact/{b}.jsonl" for b in ("olympiad", "minerva")]
    if not all(f.exists() for f in fs):
        return None
    return max(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc) for f in fs)


def extra(arm: str, s: int) -> dict:
    rs = Q.rows(arm, s)
    tr = [r["t"] - r["t_gen"] for r in rs]
    mm = [r.get("mismatch") for r in rs]
    return {"trainer_s": st.fmean(tr) if tr else None,
            "mismatch_last20": st.fmean([m for m in mm[-20:] if m is not None]) if mm else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R, steps = {"S8ML": [], "S8MS9L": []}, {}
    for arm in R:
        for s in (0, 1, 2):
            rs = Q.rows(arm, s)
            steps[f"{arm}_s{s}"] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            t = evaluated(arm, s)
            if x is None or x.get("hard_final") is None or (arm == "S8MS9L" and not (t and t <= DEADLINE)):
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            x.update(extra(arm, s), evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None)
            R[arm].append(x)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v: dict = {"steps": steps}
    if len(R["S8MS9L"]) == 3 and len(R["S8ML"]) == 3:
        f, b = R["S8MS9L"], R["S8ML"]
        v["i"] = {"met": all(steps[f"S8MS9L_s{s}"] >= 200 for s in (0, 1, 2))}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["ii"] = {"S8MS9L": h1, "S8ML": h0, "ratio": h1 / h0, "met": h1 <= 0.9 * h0}
        v["iii"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
        m1, m0 = st.fmean(x["mismatch_last20"] for x in f), st.fmean(x["mismatch_last20"] for x in b)
        v["iv"] = {"S8MS9L": m1, "S8ML": m0, "ratio": m1 / m0, "met": m1 <= 1.25 * m0}
        s0 = [x["hard_final"] for x in (Q.summary("S0G", k) for k in (0, 1, 2)) if x and x.get("hard_final") is not None]
        v["descriptive"] = {"t_gen": {arm: st.fmean(x["t_gen_mean"] for x in R[arm]) for arm in R},
                            "trainer_s": {arm: st.fmean(x["trainer_s"] for x in R[arm]) for arm in R},
                            "minus_S0G": welch([x["hard_final"] for x in f], s0) if len(s0) >= 2 else None,
                            "at_or_below_pre_rl": sum(1 for x in f if x.get("hard_base") is not None
                                                      and not x["hard_final"] > x["hard_base"])}
    else:
        v["status"] = "incomplete"
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
