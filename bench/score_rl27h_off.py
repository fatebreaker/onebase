"""RL-27B-H100-OFFLOAD (docs/REGISTRY.md, registered before any data): Qwen3.8-27B LoRA RL on one H100 with sleep mode and the
trainer's base offloaded on both arms at engine share 0.85: S8T27HO (W8A8 engine with our plugin, weight-matched trainer
through our W8A16 kernels, TIS C = 2) against S0T27HO (bf16 engine and trainer), seeds 0-2 each.
  (i)   every run completes its 100 steps
  (ii)  mean GPU-hours per run (summed logged step times), S8T27HO over S0T27HO, at most 0.90
  (iii) every run ends above its own pre-RL accuracy (bf16 engine, MATH-500+OlympiadBench+Minerva pooled)
  (iv)  Welch 95% interval of mean final MATH-500+OlympiadBench+Minerva accuracy on the bf16 engine, S8T27HO minus S0T27HO,
        within +-5 points (else inconclusive)
Stopping rule: the seeds with both arms trained and evaluated by 2026-10-22 00:00 UTC (at least two pairs; otherwise incomplete).
Descriptive: generation and trainer time per step, wake/sleep and load/offload, mismatch, TIS weights, the own-engine
evaluation, against S8T27H (resident FP8 on one H100) and RL-27B-A100-OFFLOAD.
    python3 bench/score_rl27h_off.py [--json results/std_bench/rl27h_off_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, str(LB / "bench"))
from score_rl27 import rows, summary  # noqa: E402
from score_rlmulti import welch  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 22, 0, 0, tzinfo=dt.timezone.utc)


def evaluated(name: str) -> dt.datetime | None:
    t = None
    for b in ("math500", "olympiad", "minerva"):
        f = RES / name / "bf16_exact" / f"{b}.jsonl"
        if not f.exists():
            return None
        m = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)
        t = m if t is None or m > t else t
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {"S8T27HO": {}, "S0T27HO": {}}
    steps = {}
    for arm in R:
        for k in (0, 1, 2):
            name = f"qwen27_dapo_{arm}_s{k}"
            x = summary(name, 1)
            steps[f"{arm}_s{k}"] = x["steps"]
            t = evaluated(name)
            rs = rows(name)
            x.update(seed=k, evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None, in_rule=bool(t and t <= DEADLINE),
                     t_load=st.fmean(r.get("t_load", 0.0) for r in rs) if rs else None,
                     t_offload=st.fmean(r.get("t_offload", 0.0) for r in rs) if rs else None)
            R[arm][k] = x
            print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()}))
    pairs = [k for k in (0, 1, 2) if R["S8T27HO"][k]["in_rule"] and R["S0T27HO"][k]["in_rule"]
             and R["S8T27HO"][k]["final_bf16"] is not None and R["S0T27HO"][k]["final_bf16"] is not None]
    v = {"steps": steps, "seeds_used": pairs}
    if len(pairs) < 2:
        v["status"] = "incomplete (fewer than two seed pairs trained and evaluated by the stopping date)"
    else:
        f, b = [R["S8T27HO"][k] for k in pairs], [R["S0T27HO"][k] for k in pairs]
        v["i"] = {"met": all(steps[f"{arm}_s{k}"] >= 100 for arm in R for k in pairs)}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["ii"] = {"S8T27HO": h1, "S0T27HO": h0, "ratio": h1 / h0, "met": h1 <= 0.90 * h0}
        below = [x["name"] for x in f + b if not (x["base"] is not None and x["final_bf16"] > x["base"])]
        v["iii"] = {"below": below, "met": not below}
        v["iv"] = welch([x["final_bf16"] for x in f], [x["final_bf16"] for x in b])
        v["descriptive"] = {"t_gen": {"S8T27HO": st.fmean(x["t_gen_mean"] for x in f), "S0T27HO": st.fmean(x["t_gen_mean"] for x in b)}}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": {arm: list(d.values()) for arm, d in R.items()}, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
