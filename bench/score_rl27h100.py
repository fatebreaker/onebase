"""RL-27B-H100 (docs/REGISTRY.md, registered 2026-10-19 20:23 UTC, before any data): arm S8T27H, Q27-RL's math recipe for
Qwen3.8-27B on ONE H100-80GB with vLLM's W8A8 FP8 engine and our plugin, the weight-matched trainer through our W8A16 kernels
with flash-linear-attention, TIS C = 2, no sleep mode and no offloading (train/grpo_q4.py), seeds 0-2, against Q27-RL's bf16
runs on two A100s (seeds 0-2), both scored on the bf16 engine with the exact adapter.
  (i)   every S8T27H run completes its 100 steps on one H100
  (ii)  no run ends at or below its own pre-RL accuracy (bf16 engine, MATH-500+OlympiadBench+Minerva pooled)
  (iii) Welch 95% interval of mean final MATH-500+OlympiadBench+Minerva accuracy on the bf16 engine, S8T27H minus Q27-RL,
        within +-5 points (else inconclusive)
Stopping rule: the seeds trained and evaluated by 2026-10-23 12:00 UTC (at least two). Descriptive: GPU-hours per run against
Q27-RL's (two A100s) and S8T1g's (one A100), step composition, mismatch, TIS weights, the own-engine evaluation.
    python3 bench/score_rl27h100.py [--json results/std_bench/rl27h100_verdict.json]"""
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
from score_rl27 import summary  # noqa: E402  (MATH-500+OlympiadBench+Minerva pooled; bf16_base, bf16_exact, fp8eng_exact)
from score_rlmulti import welch  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


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
    ours = []
    for k in (0, 1, 2):
        x = summary(f"qwen27_dapo_S8T27H_s{k}", 1)
        t = evaluated(x["name"])
        x.update(seed=k, evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None, in_rule=bool(t and t <= DEADLINE))
        ours.append(x)
    ref = [summary(f"qwen27_dapo_s{k}", 2) for k in (0, 1, 2)]
    t1g = [summary(f"qwen27_dapo_S8T1g_s{k}", 1) for k in (0, 1, 2)]
    for x in ours + ref:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    use = [x for x in ours if x["in_rule"] and x["final_bf16"] is not None]
    v = {"steps": [x["steps"] for x in ours], "seeds_used": [x["seed"] for x in use]}
    if len(use) < 2:
        v["status"] = "incomplete (fewer than two S8T27H runs evaluated by the stopping date)"
    else:
        # (i) is over every launched run, as registered ("every S8T27H run"), not only the evaluated ones (amendment -SCORER)
        v["i"] = {"met": all(x["steps"] >= 100 for x in ours), "steps": {x["seed"]: x["steps"] for x in ours}}
        below = [x["name"] for x in use if not (x["base"] is not None and x["final_bf16"] > x["base"])]
        v["ii"] = {"below": below, "met": not below}
        v["iii"] = welch([x["final_bf16"] for x in use], [x["final_bf16"] for x in ref if x["final_bf16"] is not None])
        v["descriptive"] = {"gpu_hours": {"S8T27H_H100": st.fmean(x["gpu_hours"] for x in use),
                                          "Q27_A100x2": st.fmean(x["gpu_hours"] for x in ref if x["gpu_hours"]),
                                          "S8T1g_A100": st.fmean(x["gpu_hours"] for x in t1g if x["gpu_hours"])}}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8T27H": ours, "Q27": ref, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
