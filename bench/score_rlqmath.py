"""RL-FP8-QMATH (docs/REGISTRY.md): Qwen3.5-9B GRPO on DAPO-Math, S0G (bf16 engine, vLLM's LoRA path, bf16 trainer) vs S8G
(online-FP8 engine with the plugin, weight-matched trainer), seeds 0-2; S4G-short (FP8 engine, unmatched trainer, 20 steps);
S9P / S9B (three jobs on one FP8 engine, pipelined / batched). Logs: <adapters>/qwen9_dapo_<arm>_s<k>/log.jsonl (single
jobs; an attempt that failed before its first checkpoint is kept as log_attempt<k>.jsonl and not read) and
<adapters>/qwen9_dapo_<S9P|S9B>/job<k>/log.jsonl; evaluations: results/std_bench/qwen9_dapo_<arm>_s<k>/<variant>/<bench>.jsonl
on each policy's own engine (S0G: bf16_base/bf16_exact; the FP8 arms: fp8eng_base/fp8eng_exact).
  (i)   S8G's mean |delta log p| over steps 1-10 below S4G's at the same seed, in every seed
  (ii)  S8G's mean generation time per step at most 0.8x S0G's (means over seeds)
  (iii) GPU-hours per run of S9P and of S9B each at most 0.8x S8G's mean
  (iv)  Welch 95% interval of mean final accuracy on OlympiadBench+Minerva pooled, S8G - S0G, within +-5 (else inconclusive);
        also S9P - S0G and S9B - S0G
Descriptive: MATH-500, AIME24/25 and AMC23 avg@16, rewards, gains over each engine's base, GPU-hours.
    python3 bench/score_rlqmath.py [--json results/std_bench/rlqmath_verdict.json]"""
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

EV = {"S0G": ("bf16_base", "bf16_exact"), "S8G": ("fp8eng_base", "fp8eng_exact"), "S9P": ("fp8eng_base", "fp8eng_exact"),
      "S9B": ("fp8eng_base", "fp8eng_exact")}


def log_file(arm: str, s: int) -> Path:
    if arm in ("S9P", "S9B"):
        return AD / f"qwen9_dapo_{arm}/job{s}/log.jsonl"
    return AD / f"qwen9_dapo_{arm}_s{s}/log.jsonl"              # the trainer's own log (a failed attempt's is set aside)


def rows(arm: str, s: int) -> list[dict]:
    """One row per step (bench/runlog.py: a resumed run's repeated steps are read from the attempt it continued)."""
    return runlog.read(log_file(arm, s))


def acc(arm: str, s: int, variant: str, benches: tuple[str, ...]) -> float | None:
    vals = []
    for b in benches:
        f = RES / f"qwen9_dapo_{arm}_s{s}/{variant}/{b}.jsonl"
        if not f.exists():
            return None
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines()]
    return 100 * sum(vals) / len(vals)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summary(arm: str, s: int, steps: int = 200) -> dict | None:
    rs = rows(arm, s)
    if not rs or rs[-1]["step"] < steps:
        return None
    base, ex = EV.get(arm, (None, None))
    out = {"arm": arm, "seed": s, "mismatch_first10": mean([r.get("mismatch") for r in rs if r["step"] <= 10]),
           "t_gen_mean": mean([r["t_gen"] for r in rs]), "gpu_hours": sum(r["t"] for r in rs) / 3600,
           "gpu_hours_lost": runlog.lost_hours(log_file(arm, s)),
           "reward_early": mean([r["reward"] for r in rs if r["step"] <= 50]),
           "reward_late": mean([r["reward"] for r in rs if r["step"] > steps - 50])}
    if base:
        for lab, bs in (("hard", ("olympiad", "minerva")), ("math500", ("math500",)), ("aime_amc", ("aime24", "aime25", "amc23"))):
            out[f"{lab}_base"], out[f"{lab}_final"] = acc(arm, s, base, bs), acc(arm, s, ex, bs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: [x for x in (summary(arm, s) for s in a.seeds) if x] for arm in ("S0G", "S8G", "S9P", "S9B")}
    R["S4G"] = [x for x in (summary("S4G", s, steps=20) for s in a.seeds) if x]
    for arm, xs in R.items():
        for x in xs:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {}
    s8 = {x["seed"]: x for x in R["S8G"]}
    s4 = {x["seed"]: x for x in R["S4G"]}
    common = sorted(set(s8) & set(s4))
    if common and len(common) == len(a.seeds):
        v["i"] = {"S8G": [s8[k]["mismatch_first10"] for k in common], "S4G": [s4[k]["mismatch_first10"] for k in common],
                  "verdict": "MET" if all(s8[k]["mismatch_first10"] < s4[k]["mismatch_first10"] for k in common) else "NOT MET"}
    if len(R["S0G"]) == len(a.seeds) and len(R["S8G"]) == len(a.seeds):
        t0, t8 = mean([x["t_gen_mean"] for x in R["S0G"]]), mean([x["t_gen_mean"] for x in R["S8G"]])
        v["ii"] = {"S0G": t0, "S8G": t8, "ratio": t8 / t0, "verdict": "MET" if t8 <= 0.8 * t0 else "NOT MET"}
        h8 = mean([x["gpu_hours"] for x in R["S8G"]])
        for arm in ("S9P", "S9B"):
            if len(R[arm]) == len(a.seeds):
                h = mean([x["gpu_hours"] for x in R[arm]])
                v[f"iii_{arm}"] = {"gpu_hours": h, "S8G": h8, "ratio": h / h8, "verdict": "MET" if h <= 0.8 * h8 else "NOT MET"}
        y = [x["hard_final"] for x in R["S0G"] if x.get("hard_final") is not None]
        for arm in ("S8G", "S9P", "S9B"):
            x_ = [x["hard_final"] for x in R[arm] if x.get("hard_final") is not None]
            if len(x_) >= 2 and len(y) >= 2:
                v[f"iv_{arm}"] = welch(x_, y)
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
