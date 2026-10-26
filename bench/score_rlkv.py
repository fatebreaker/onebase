"""RL-KVSLEEP (docs/REGISTRY.md; this scorer written before any data): one shared FP8 copy whose in-process engine releases only its KV
cache while the trainer runs (train/grpo_q16.py --kv-sleep), at each offload arm's engine share.
  arm S8K9L  (Qwen3.5-9B,  L40S) against S8OL   (RL-L40S-OFFLOAD: sleep mode, the trainer's base offloaded, share 0.80)
  arm S8K27L (Qwen3.8-27B, L40S) against S8T27L (RL-27B-L40S: the same, share 0.85)
  (i)   every run completes its steps (200; 100 for the 27B)
  (ii)  each arm's mean GPU-hours per run at most 1.00x its offload arm's
  (iii) final accuracy (OlympiadBench+Minerva pooled, own FP8 engine), each arm minus its offload arm: Welch 95% interval within +-5
        points, else inconclusive
Runs trained and evaluated by 2026-10-26 12:00 UTC count.
    python3 bench/score_rlkv.py [--json results/std_bench/rlkv_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_rlmulti import welch  # noqa: E402

DEADLINE = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.timezone.utc)
AD = Path("$SCRATCH/std_bench/adapters")
RES = LB / "results/std_bench"
PAIRS = (("S8K9L", "S8OL", "qwen9", 200, (0, 1, 2)), ("S8K27L", "S8T27L", "qwen27", 100, (0, 1, 2)))


def run(prefix: str, arm: str, s: int, steps: int) -> dict | None:
    name = f"{prefix}_dapo_{arm}_s{s}"
    rows = runlog.read(AD / name / "log.jsonl")
    if not rows or rows[-1]["step"] < steps:
        return {"name": name, "steps": rows[-1]["step"] if rows else 0, "complete": False}
    vals, times = [], []
    for b in ("olympiad", "minerva"):
        f = RES / name / "fp8eng_exact" / f"{b}.jsonl"
        if not f.exists():
            return {"name": name, "steps": rows[-1]["step"], "complete": True, "evaluated": False}
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines()]
        times.append(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc))
    return {"name": name, "steps": rows[-1]["step"], "complete": True, "evaluated": True, "evaluated_utc": max(times),
            "final": 100 * sum(vals) / len(vals), "gpu_hours": sum(r["t"] for r in rows) / 3600,
            "t_gen": st.fmean(r["t_gen"] for r in rows), "t": st.fmean(r["t"] for r in rows),
            "t_wake": st.fmean(r.get("t_wake", 0.0) for r in rows), "t_sleep": st.fmean(r.get("t_sleep", 0.0) for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out: dict = {}
    for arm, ref, prefix, steps, seeds in PAIRS:
        xs = [run(prefix, arm, s, steps) for s in seeds]
        rs = [run(prefix, ref, s, steps) for s in seeds]
        ok = [x for x in xs if x and x.get("evaluated") and x["evaluated_utc"] <= DEADLINE]
        okr = [x for x in rs if x and x.get("evaluated")]
        v: dict = {"steps": {x["name"]: x["steps"] for x in xs if x}}
        if len(ok) == len(seeds) and len(okr) >= 2:
            v["i"] = {"met": all(x["steps"] >= steps for x in ok)}
            h1, h0 = st.fmean(x["gpu_hours"] for x in ok), st.fmean(x["gpu_hours"] for x in okr)
            v["ii"] = {arm: h1, ref: h0, "ratio": h1 / h0, "met": h1 <= h0}
            v["iii"] = welch([x["final"] for x in ok], [x["final"] for x in okr])
            v["descriptive"] = {k: {arm: st.fmean(x[k] for x in ok), ref: st.fmean(x[k] for x in okr)}
                                for k in ("t_gen", "t", "t_wake", "t_sleep")}
        else:
            v["status"] = "incomplete"
        out[arm] = v
        print(arm, json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
