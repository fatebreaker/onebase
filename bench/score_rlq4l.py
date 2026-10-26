"""RL-FP8-Q4L (docs/REGISTRY.md): Qwen3-4B-Instruct-2507 GRPO on DAPO-Math on an L40S, S0L (bf16 engine, vLLM's LoRA path,
bf16 trainer) vs S8L (online-FP8 engine with the plugin, weight-matched trainer), seeds 0-2 alternating; the analysis
uses the seeds whose two arms both finished training and evaluation by 2026-10-21 00:00 UTC (at least two).
  (i)   S8L's mean generation time per step at most 0.8x S0L's (means over the seeds used)
  (ii)  S8L's GPU-hours per run (summed logged step times) at most 0.8x S0L's
  (iii) Welch 95% interval of mean final accuracy on OlympiadBench+Minerva pooled, S8L - S0L, within +-5 (else
        inconclusive), each policy on its own engine (S0L: bf16_exact; S8L: fp8eng_exact through the plugin)
Descriptive: trainer-engine mismatch over steps 1-10, rewards, MATH-500, each arm's gain over its base.
Logs: <adapters>/qwen4_dapo_<arm>_s<k>/log.jsonl; evaluations: results/std_bench/qwen4_dapo_<arm>_s<k>/<variant>/<bench>.jsonl.
    python3 bench/score_rlq4l.py [--json results/std_bench/rlq4l_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
AD = Path("$SCRATCH/std_bench/adapters")
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_rlmulti import welch  # noqa: E402

EV = {"S0L": ("bf16_base", "bf16_exact"), "S8L": ("fp8eng_base", "fp8eng_exact"), "S9PL": ("fp8eng_base", "fp8eng_exact")}
STEPS = 100


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def acc(arm, s, variant, benches):
    vals = []
    for b in benches:
        f = RES / f"qwen4_dapo_{arm}_s{s}/{variant}/{b}.jsonl"
        if not f.exists():
            return None
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines()]
    return 100 * sum(vals) / len(vals)


def summary(arm, s):
    f = AD / (f"qwen4_dapo_S9PL/job{s}/log.jsonl" if arm == "S9PL" else f"qwen4_dapo_{arm}_s{s}/log.jsonl")  # S9PL: job k = seed k
    rs = runlog.read(f)                                       # one row per step (bench/runlog.py)
    if not rs or rs[-1]["step"] < STEPS:
        return None
    base, ex = EV[arm]
    out = {"arm": arm, "seed": s, "mismatch_first10": mean([r.get("mismatch") for r in rs if r["step"] <= 10]),
           "t_gen_mean": mean([r["t_gen"] for r in rs]), "gpu_hours": sum(r["t"] for r in rs) / 3600,
           "reward_early": mean([r["reward"] for r in rs if r["step"] <= 25]),
           "reward_late": mean([r["reward"] for r in rs if r["step"] > STEPS - 25])}
    for lab, bs in (("hard", ("olympiad", "minerva")), ("math500", ("math500",))):
        out[f"{lab}_base"], out[f"{lab}_final"] = acc(arm, s, base, bs), acc(arm, s, ex, bs)
    return out if out["hard_final"] is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: {s: summary(arm, s) for s in (0, 1, 2)} for arm in EV}
    seeds = [s for s in (0, 1, 2) if R["S0L"][s] and R["S8L"][s]]
    multi = [s for s in (0, 1, 2) if R["S9PL"][s]]
    for arm in EV:
        for s in seeds:
            if R[arm][s]:
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in R[arm][s].items()}))
    v = {"seeds": seeds}
    if len(seeds) >= 2:
        t0, t8 = (mean([R[arm][s]["t_gen_mean"] for s in seeds]) for arm in ("S0L", "S8L"))
        h0, h8 = (mean([R[arm][s]["gpu_hours"] for s in seeds]) for arm in ("S0L", "S8L"))
        v["i"] = {"S0L": t0, "S8L": t8, "ratio": t8 / t0, "verdict": "MET" if t8 <= 0.8 * t0 else "NOT MET"}
        v["ii"] = {"S0L": h0, "S8L": h8, "ratio": h8 / h0, "verdict": "MET" if h8 <= 0.8 * h0 else "NOT MET"}
        v["iii"] = welch([R["S8L"][s]["hard_final"] for s in seeds], [R["S0L"][s]["hard_final"] for s in seeds])
    else:
        v["note"] = "fewer than two seeds with both arms complete"
    if len(multi) == 3 and len(seeds) == 3:                    # RL-FP8-Q4L-MULTI (registered 2026-10-16 11:38 UTC)
        hm = mean([R["S9PL"][s]["gpu_hours"] for s in multi])
        h8 = mean([R["S8L"][s]["gpu_hours"] for s in seeds])
        v["multi_i"] = {"S9PL": hm, "S8L": h8, "ratio": hm / h8, "verdict": "MET" if hm <= 0.8 * h8 else "NOT MET"}
        v["multi_ii"] = welch([R["S9PL"][s]["hard_final"] for s in multi], [R["S0L"][s]["hard_final"] for s in seeds])
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
