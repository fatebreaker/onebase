"""RL-SHARED (docs/REGISTRY.md, registered 2026-10-21 20:58 UTC, before any data): the 27B's math RL with ONE FP8 copy of the
base mapped by the trainer from the rollout engine (train/grpo_q14.py --shared-base), no sleep mode and no offloading.
  S8S27L  one 48 GB L40S, S8T27L's recipe without --sleep --offload-base (scored as S8T27L: bench/score_rl27l40s.summary,
          OlympiadBench+Minerva pooled on its own FP8 engine and on the bf16 engine)
  S8S27H  one H100-80GB, S8T27H's recipe, engine share 0.65 (scored as S8T27H: bench/score_rl27.summary, MATH-500+OlympiadBench+
          Minerva pooled on the bf16 engine)
  (i)   every S8S27L run completes its 100 steps          (ii)  every S8S27H run completes its 100 steps
  (iii) S8S27H's mean GPU-hours per run at most 0.9x S8T27H's
  (iv)  S8S27L minus Q27-RL (the same two benchmarks, bf16 engine) and S8S27H minus Q27-RL (three benchmarks), each a Welch 95%
        interval within +-5 points (else inconclusive)
Stopping rule: runs trained and evaluated by 2026-10-24 12:00 UTC, at least two per arm (else that arm is incomplete).
Descriptive: GPU-hours against S8T27L (offload, L40S), S8T27HO and S0T27HO (offload, H100); step time, generation time,
mismatch, TIS weights.
    python3 bench/score_rlshared.py [--json results/std_bench/rlshared_verdict.json]"""
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
import score_rl27 as R27  # noqa: E402  (three benchmarks, bf16 engine)
import score_rl27l40s as R27L  # noqa: E402  (OlympiadBench+Minerva, own and bf16 engines)
from score_rlmulti import welch  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 24, 12, 0, tzinfo=dt.timezone.utc)


def evaluated3(name: str) -> dt.datetime | None:
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
    sl = []
    for k in (0, 1, 2):
        x = R27L.summary(f"qwen27_dapo_S8S27L_s{k}", 1)
        x["in_rule"] = bool(x["evaluated_utc"]) and dt.datetime.fromisoformat(x["evaluated_utc"]).replace(tzinfo=dt.timezone.utc) <= DEADLINE
        x["seed"] = k
        sl.append(x)
    sh = []
    for k in (0, 1, 2):
        x = R27.summary(f"qwen27_dapo_S8S27H_s{k}", 1)
        t = evaluated3(x["name"])
        x.update(seed=k, evaluated_utc=t.strftime("%Y-%m-%dT%H:%M") if t else None, in_rule=bool(t and t <= DEADLINE))
        sh.append(x)
    q_hard = [R27L.acc(f"qwen27_dapo_s{k}", "bf16_exact")[0] for k in (0, 1, 2)]
    q_three = [R27.summary(f"qwen27_dapo_s{k}", 2)["final_bf16"] for k in (0, 1, 2)]
    def done_hours(fn, arm):            # completed runs only (S8T27H's seed 2 stopped at step 87; the paper's 4.4)
        return [x["gpu_hours"] for x in (fn(f"qwen27_dapo_{arm}_s{k}", 1) for k in (0, 1, 2)) if x["steps"] >= 100]
    comp = {"S8T27L_L40S": done_hours(R27L.summary, "S8T27L"), "S8T27H_H100": done_hours(R27.summary, "S8T27H"),
            "S8T27HO_H100": done_hours(R27.summary, "S8T27HO"), "S0T27HO_H100": done_hours(R27.summary, "S0T27HO")}
    comp = {k: st.fmean(v) for k, v in ((k, [x for x in v if x]) for k, v in comp.items()) if v}
    for x in sl + sh:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {"steps": {"S8S27L": [x["steps"] for x in sl], "S8S27H": [x["steps"] for x in sh]}, "comparators_gpu_hours": comp}
    ul = [x for x in sl if x["in_rule"] and x["final_bf16"] is not None]
    uh = [x for x in sh if x["in_rule"] and x["final_bf16"] is not None]
    if len(ul) >= 2:
        v["i"] = {"met": all(x["steps"] >= 100 for x in sl if x["steps"]), "steps": [x["steps"] for x in sl]}
        v["iv_L"] = welch([x["final_bf16"] for x in ul], [q for q in q_hard if q is not None])
        v["gpu_hours_L"] = st.fmean(x["gpu_hours"] for x in ul)
    else:
        v["status_L"] = "incomplete (fewer than two S8S27L runs evaluated by the stopping date)"
    if len(uh) >= 2:
        v["ii"] = {"met": all(x["steps"] >= 100 for x in sh if x["steps"]), "steps": [x["steps"] for x in sh]}
        h = st.fmean(x["gpu_hours"] for x in uh)
        v["iii"] = {"S8S27H": h, "S8T27H": comp.get("S8T27H_H100"), "ratio": h / comp["S8T27H_H100"] if comp.get("S8T27H_H100") else None,
                    "met": bool(comp.get("S8T27H_H100")) and h <= 0.9 * comp["S8T27H_H100"]}
        v["iv_H"] = welch([x["final_bf16"] for x in uh], [q for q in q_three if q is not None])
    else:
        v["status_H"] = "incomplete (fewer than two S8S27H runs evaluated by the stopping date)"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8S27L": sl, "S8S27H": sh, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
