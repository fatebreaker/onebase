"""RL-27B-1GPU (docs/REGISTRY.md, registered 2026-10-17): arm S8T1g, Q27-RL's math recipe on ONE GPU (FP8 engine, weight-matched
trainer, TIS), seeds 0-2, against Q27-RL's bf16 runs on two GPUs (engine and trainer on one A100 each), seeds 0-2.
  (i)   every S8T1g run completes its 100 steps
  (ii)  mean GPU-hours per run (summed logged step times x GPUs) at most 0.9x Q27-RL's
  (iii) mean final accuracy on MATH-500+OlympiadBench+Minerva pooled, both evaluated on the bf16 engine with the exact adapter,
        S8T1g minus Q27-RL: Welch 95% interval within +-5 points (else inconclusive)
Descriptive: step and generation times, mismatch, TIS weights, the own-engine (FP8) evaluation.
Logs <adapters>/qwen27_dapo_<S8T1g_s|s><k>/log.jsonl; evaluations results/std_bench/qwen27_dapo_<...>/<variant>/<bench>.jsonl.
    python3 bench/score_rl27.py [--json results/std_bench/rl27_verdict.json]"""
from __future__ import annotations

import argparse
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

BENCHES = ("math500", "olympiad", "minerva")


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def acc(name: str, variant: str) -> float | None:
    v = []
    for b in BENCHES:
        f = RES / name / variant / f"{b}.jsonl"
        if not f.exists():
            return None
        v += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines()]
    return 100 * sum(v) / len(v)


def summary(name: str, gpus: int) -> dict:
    rs = rows(name)
    mm = [r["mismatch"] for r in rs if r.get("mismatch") is not None]
    return {"name": name, "steps": rs[-1]["step"] if rs else 0, "gpu_hours": gpus * sum(r["t"] for r in rs) / 3600 if rs else None,
            "t_mean": st.fmean(r["t"] for r in rs) if rs else None, "t_gen_mean": st.fmean(r["t_gen"] for r in rs) if rs else None,
            "mismatch_mean": st.fmean(mm) if mm else None, "tis_mean": st.fmean(r["tis_mean"] for r in rs if r.get("tis_mean")) if any(r.get("tis_mean") for r in rs) else None,
            "base": acc(name, "bf16_base"), "final_bf16": acc(name, "bf16_exact"), "final_own": acc(name, "fp8eng_exact")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ours = [summary(f"qwen27_dapo_S8T1g_s{k}", 1) for k in (0, 1, 2)]
    ref = [summary(f"qwen27_dapo_s{k}", 2) for k in (0, 1, 2)]
    for x in ours + ref:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {}
    done = [x for x in ours if x["final_bf16"] is not None]
    if len(done) == 3:
        v["i"] = {"steps": [x["steps"] for x in ours], "met": all(x["steps"] >= 100 for x in ours)}
        h1, h2 = st.fmean(x["gpu_hours"] for x in ours), st.fmean(x["gpu_hours"] for x in ref)
        v["ii"] = {"S8T1g": h1, "Q27_bf16_2gpu": h2, "ratio": h1 / h2, "met": h1 <= 0.9 * h2}
        v["iii"] = welch([x["final_bf16"] for x in ours], [x["final_bf16"] for x in ref])
    else:
        v["status"] = f"{len(done)} of 3 S8T1g runs evaluated"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8T1g": ours, "Q27": ref, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
