"""RL-27B-L40S (docs/REGISTRY.md, registered 2026-10-19 05:36 UTC, before any data): arm S8T27L, the 27B's LoRA RL on ONE 48 GB
L40S with FP8 on both sides (train/grpo_q9.py since amendment -Q9, before any data; the trainer's base offloaded during rollouts),
seeds 0-2, against Q27-RL's bf16
runs on two A100s (seeds 0-2), both scored on the bf16 engine with the exact adapter.
  (i)   every S8T27L run completes its 100 steps on one L40S
  (ii)  no run ends at or below its own pre-RL accuracy (own FP8 engine, OlympiadBench+Minerva pooled)
  (iii) Welch 95% interval of mean final OlympiadBench+Minerva accuracy on the bf16 engine, S8T27L minus Q27-RL, within +-5
        points (else inconclusive)
Stopping rule: the seeds trained and evaluated on both engines by 2026-10-23 12:00 UTC (at least two). Descriptive: L40S-hours
per run against Q27-RL's A100-hours (two GPUs) and S8T1g's (one A100), step composition, mismatch, TIS weights.
Logs <adapters>/qwen27_dapo_S8T27L_s<k>/log.jsonl; evaluations results/std_bench/qwen27_dapo_S8T27L_s<k>/<variant>/<bench>.jsonl
(fp8eng_base, fp8eng_exact on the L40S; bf16_exact on an A100).
    python3 bench/score_rl27l40s.py [--json results/std_bench/rl27l40s_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
AD = Path("$SCRATCH/std_bench/adapters")
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_rlmulti import welch  # noqa: E402

HARD = ("olympiad", "minerva")
DEADLINE = dt.datetime(2026, 10, 23, 12, 0, tzinfo=dt.timezone.utc)


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def acc(name: str, variant: str) -> tuple[float | None, dt.datetime | None]:
    v, t = [], None
    for b in HARD:
        f = RES / name / variant / f"{b}.jsonl"
        if not f.exists():
            return None, None
        v += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines()]
        m = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc)
        t = m if t is None or m > t else t
    return 100 * sum(v) / len(v), t


def summary(name: str, gpus: int) -> dict:
    rs = rows(name)
    mm = [r["mismatch"] for r in rs if r.get("mismatch") is not None]
    base, _ = acc(name, "fp8eng_base")
    own, t_own = acc(name, "fp8eng_exact")
    bf, t_bf = acc(name, "bf16_exact")
    return {"name": name, "steps": rs[-1]["step"] if rs else 0, "gpu_hours": gpus * sum(r["t"] for r in rs) / 3600 if rs else None,
            "t_mean": st.fmean(r["t"] for r in rs) if rs else None, "t_gen_mean": st.fmean(r["t_gen"] for r in rs) if rs else None,
            "mismatch_mean": st.fmean(mm) if mm else None,
            "tis_mean": st.fmean(r["tis_mean"] for r in rs if r.get("tis_mean")) if any(r.get("tis_mean") for r in rs) else None,
            "base_own": base, "final_own": own, "final_bf16": bf,
            "evaluated_utc": max(t for t in (t_own, t_bf) if t).strftime("%Y-%m-%dT%H:%M") if t_own and t_bf else None,
            "in_rule": bool(t_own and t_bf and max(t_own, t_bf) <= DEADLINE)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ours = [summary(f"qwen27_dapo_S8T27L_s{k}", 1) for k in (0, 1, 2)]
    ref = []
    for k in (0, 1, 2):                                  # Q27-RL: bf16 on two GPUs, scored on the bf16 engine
        x = summary(f"qwen27_dapo_s{k}", 2)
        x["final_bf16"], _ = acc(f"qwen27_dapo_s{k}", "bf16_exact")
        ref.append(x)
    for x in ours + ref:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    use = [x for x in ours if x["in_rule"]]
    v = {"steps": [x["steps"] for x in ours], "seeds_used": [x["name"][-1] for x in use]}
    if len(use) < 2:
        v["status"] = "incomplete (fewer than two S8T27L runs evaluated on both engines by the stopping date)"
    else:
        v["i"] = {"met": all(x["steps"] >= 100 for x in use)}
        below = [x["name"] for x in use if not (x["final_own"] is not None and x["base_own"] is not None and x["final_own"] > x["base_own"])]
        v["ii"] = {"below": below, "met": not below}
        v["iii"] = welch([x["final_bf16"] for x in use], [x["final_bf16"] for x in ref if x["final_bf16"] is not None])
        v["gpu_hours"] = {"S8T27L_L40S": st.fmean(x["gpu_hours"] for x in use),
                          "Q27_A100x2": st.fmean(x["gpu_hours"] for x in ref if x["gpu_hours"])}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8T27L": ours, "Q27": ref, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
