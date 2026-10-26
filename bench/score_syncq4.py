"""SYNC-Q4 (docs/REGISTRY.md): Qwen3-4B-Instruct-2507 GRPO on DAPO-Math, exact syncs (A_*) vs merge-syncs (B_*) into bf16,
FP8 and NVFP4 engines, seed 0. Final policies (the trainer's adapter) served exactly on their own base, MATH-500,
OlympiadBench and Minerva pooled (1,447 problems).
  (i)   delivered share: B_nv at most 0.05 at every sync; B_f8's median below 0.2; B_bf's mean over syncs 1-10 below 0.5
  (ii)  the exact arms learn: mean rollout reward over steps 76-100 exceeds steps 1-25 by more than 0.03 (A_bf, A_f8, A_nv)
  (iii) on the FP8 and the NVFP4 base, the exact arm's final policy beats the merge arm's (McNemar p <= 0.05)
Descriptive: every arm's final accuracy, the bf16 pair, mismatch over steps 1-10 and 91-100, the collapse step (10-step
mean reward below half its running maximum).
Files: <adapters>/qwen4_dapo_<arm>_s0/log.jsonl; results/std_bench/qwen4_dapo_<arm>_s0/<base>_exact/<bench>.jsonl.
    python3 bench/score_syncq4.py [--seed k] [--json results/std_bench/syncq4_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
import runlog  # noqa: E402

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
AD = Path("$SCRATCH/std_bench/adapters")
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

BASE = {"A_bf": "bf16", "B_bf": "bf16", "A_f8": "fp8", "B_f8": "fp8", "A_nv": "nvfp4", "B_nv": "nvfp4"}
BENCHES = ("math500", "olympiad", "minerva")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def log(arm: str, seed: int = 0) -> list[dict]:
    f = AD / f"qwen4_dapo_{arm}_s{seed}/log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def acc(arm: str, seed: int = 0) -> dict | None:
    out = {}
    for b in BENCHES:
        f = RES / f"qwen4_dapo_{arm}_s{seed}/{BASE[arm]}_exact/{b}.jsonl"
        if not f.exists():
            return None
        out.update({f"{b}:{r['id']}": int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())})
    return out


def collapse_step(rows):
    best = 0.0
    for t in range(10, len(rows) + 1):
        m = mean([r["reward"] for r in rows[t - 10:t]])
        best = max(best, m)
        if m < 0.5 * best:
            return rows[t - 1]["step"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=0, help="SYNC-Q4-SEEDS: seeds 1-2 have the four low-precision arms only")
    a = ap.parse_args()
    L = {arm: log(arm, a.seed) for arm in BASE}
    done = {arm: bool(L[arm]) and L[arm][-1]["step"] >= 100 for arm in BASE}
    A = {arm: acc(arm, a.seed) for arm in BASE}
    out = {"arms": {}, "verdict": {}}
    for arm in BASE:
        rs = L[arm]
        if not rs:
            continue
        g = [r["sync_gain"] for r in rs if r.get("sync_gain") is not None]
        e = {"steps": rs[-1]["step"], "reward_1_25": mean([r["reward"] for r in rs if r["step"] <= 25]),
             "reward_76_100": mean([r["reward"] for r in rs if 76 <= r["step"] <= 100]),
             "mismatch_1_10": mean([r.get("mismatch") for r in rs if r["step"] <= 10]),
             "mismatch_91_100": mean([r.get("mismatch") for r in rs if 91 <= r["step"] <= 100]),
             "collapse_step": collapse_step(rs)}
        if g:
            e |= {"delivered_max": max(g), "delivered_median": statistics.median(g),
                  "delivered_1_10": mean([r["sync_gain"] for r in rs if r["step"] <= 10 and r.get("sync_gain") is not None])}
        if A[arm]:
            e["acc"] = 100 * sum(A[arm].values()) / len(A[arm])
        out["arms"][arm] = e
        print(arm, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}))
    v = out["verdict"]
    merges = ("B_bf", "B_f8", "B_nv") if a.seed == 0 else ("B_f8", "B_nv")
    if all(done[x] for x in merges):
        ar = out["arms"]
        v["i"] = {"B_nv_max": ar["B_nv"]["delivered_max"], "B_f8_median": ar["B_f8"]["delivered_median"],
                  "met": ar["B_nv"]["delivered_max"] <= 0.05 and ar["B_f8"]["delivered_median"] < 0.2}
        if a.seed == 0:
            v["i"]["B_bf_mean_1_10"] = ar["B_bf"]["delivered_1_10"]
            v["i"]["met"] = v["i"]["met"] and ar["B_bf"]["delivered_1_10"] < 0.5
    exacts = ("A_bf", "A_f8", "A_nv") if a.seed == 0 else ("A_f8", "A_nv")
    if all(done[x] for x in exacts):
        ar = out["arms"]
        gains = {x: ar[x]["reward_76_100"] - ar[x]["reward_1_25"] for x in exacts}
        v["ii"] = {"gains": gains, "met": all(g > 0.03 for g in gains.values())}
    for base, ea, ma in (("fp8", "A_f8", "B_f8"), ("nvfp4", "A_nv", "B_nv")):
        if A[ea] and A[ma]:
            ids = sorted(set(A[ea]) & set(A[ma]))
            w, l, p = rr.mcnemar([A[ea][i] for i in ids], [A[ma][i] for i in ids])
            v[f"iii_{base}"] = {"exact_vs_merge": [w, l, p], "met": w > l and p <= 0.05}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
