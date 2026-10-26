"""RL-FP8-MATH (docs/REGISTRY.md): LoRA RL on MATH with Llama-3.1-8B-Instruct, S0 (bf16 engine, vLLM's LoRA path, bf16
trainer) against S8 (online-FP8 engine with the plugin, weight-matched trainer), seeds 0-2. Reads each run's training
log (logs/lb_rlmath_llama_math_<arm>_s<seed>.out) and its MATH-500 evaluation on its own engine
(results/std_bench/llama_math_<arm>_s<seed>/{bf16,fp8eng}_{base,exact}/math500.jsonl).
  (i)   mean trainer-engine |delta log p| over steps 1-10: S0 < 0.01, S8 < 0.02
  (ii)  S8's mean generation time per step <= 0.8 x S0's
  (iii) Welch 95% interval of mean final MATH-500 accuracy, S8 - S0, within +-5 points (else inconclusive)
Descriptive: rewards, each arm's gain over its base, GPU-hours per run (sum of step times).
Amendment RL-FP8-MATH-G (CUDA-graph engines): S0G and S8G judged by (i), (ii) and (iii) as S0 and S8, and (iii) pooled over
all six runs per system (S0+S0G vs S8+S8G). --arms G for those.
    python3 bench/score_rlmath.py [--json results/std_bench/rlmath_verdict.json]"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
EV = {"S0": ("bf16_base", "bf16_exact"), "S8": ("fp8eng_base", "fp8eng_exact"),
      "S0G": ("bf16_base", "bf16_exact"), "S8G": ("fp8eng_base", "fp8eng_exact")}


def acc(f: Path) -> float | None:
    if not f.exists():
        return None
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    return 100 * sum(int(r["correct"]) for r in rows) / len(rows)


def t_quantile(p: float, df: float) -> float:
    """Student t quantile by bisection on the regularized incomplete beta (no scipy on the login node)."""
    def cdf(t: float) -> float:
        x = df / (df + t * t)
        # regularized incomplete beta I_x(df/2, 1/2) by continued fraction (Numerical Recipes betacf)
        a, b = df / 2, 0.5
        bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
        def cf(a, b, x):
            qab, qap, qam = a + b, a + 1, a - 1
            c, d = 1.0, 1 - qab * x / qap
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            h = d
            for m in range(1, 200):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
                h *= d * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                d = 1 + aa * d; d = 1 / (d if abs(d) > 1e-30 else 1e-30)
                c = 1 + aa / c if abs(1 + aa / c) > 1e-30 else 1e-30
                h *= d * c
            return h
        ib = bt * cf(a, b, x) / a
        return 1 - 0.5 * ib if t > 0 else 0.5 * ib
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(arm: str, seed: int) -> dict | None:
    name = f"llama_math_{arm}_s{seed}"
    f = LB / f"logs/lb_rlmath_{name}.out"
    rows = [json.loads(x) for x in f.read_text().splitlines() if x.startswith('{"step"')] if f.exists() else []
    if not rows or rows[-1]["step"] < 200:
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    base, exact = EV[arm]
    return {"name": name,
            "mismatch_first10": mean([r["mismatch"] for r in rows if r["step"] <= 10 and r.get("mismatch") is not None]),
            "t_gen_mean": mean([r["t_gen"] for r in rows]),
            "gpu_hours": sum(r["t"] for r in rows) / 3600,
            "reward_early": mean([r["reward"] for r in rows if r["step"] <= 50]),
            "reward_late": mean([r["reward"] for r in rows if r["step"] > 150]),
            "acc_base": acc(RES / name / base / "math500.jsonl"),
            "acc_final": acc(RES / name / exact / "math500.jsonl")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--json", default="")
    ap.add_argument("--arms", default="eager", choices=("eager", "G", "pooled"))
    a = ap.parse_args()
    if a.arms == "pooled":                                       # (iii) over all six runs per system
        res = {"S0": [r for arm in ("S0", "S0G") for r in (run(arm, s) for s in a.seeds) if r],
               "S8": [r for arm in ("S8", "S8G") for r in (run(arm, s) for s in a.seeds) if r]}
    else:
        z, e = ("S0", "S8") if a.arms == "eager" else ("S0G", "S8G")
        res = {"S0": [r for r in (run(z, s) for s in a.seeds) if r], "S8": [r for r in (run(e, s) for s in a.seeds) if r]}
    for arm, rs in res.items():
        for r in rs:
            print(f"{r['name']}: mismatch(1-10) {r['mismatch_first10']:.4f}; t_gen {r['t_gen_mean']:.1f}s; GPU-h {r['gpu_hours']:.2f}; "
                  f"reward {r['reward_early']:.3f}->{r['reward_late']:.3f}; MATH-500 base {r['acc_base']} final {r['acc_final']}")
    v = {}
    if res["S0"] and res["S8"]:
        m0 = sum(r["mismatch_first10"] for r in res["S0"]) / len(res["S0"])
        m8 = sum(r["mismatch_first10"] for r in res["S8"]) / len(res["S8"])
        v["i"] = {"S0": m0, "S8": m8, "verdict": "MET" if m0 < 0.01 and m8 < 0.02 else "NOT MET"}
        t0 = sum(r["t_gen_mean"] for r in res["S0"]) / len(res["S0"])
        t8 = sum(r["t_gen_mean"] for r in res["S8"]) / len(res["S8"])
        v["ii"] = {"S0": t0, "S8": t8, "ratio": t8 / t0, "verdict": "MET" if t8 <= 0.8 * t0 else "NOT MET"}
        x = [r["acc_final"] for r in res["S8"] if r["acc_final"] is not None]
        y = [r["acc_final"] for r in res["S0"] if r["acc_final"] is not None]
        if len(x) >= 2 and len(y) >= 2:
            mx, my = sum(x) / len(x), sum(y) / len(y)
            vx = sum((t - mx) ** 2 for t in x) / (len(x) - 1)
            vy = sum((t - my) ** 2 for t in y) / (len(y) - 1)
            se = math.sqrt(vx / len(x) + vy / len(y))
            dfw = se ** 4 / ((vx / len(x)) ** 2 / (len(x) - 1) + (vy / len(y)) ** 2 / (len(y) - 1)) if se > 0 else 1e9
            q = t_quantile(0.975, dfw) if se > 0 else 0.0
            lo, hi = mx - my - q * se, mx - my + q * se
            v["iii"] = {"diff": mx - my, "ci95": [lo, hi], "df": dfw, "n": [len(x), len(y)],
                        "verdict": "MET" if -5 <= lo and hi <= 5 else ("INCONCLUSIVE" if lo < 5 and hi > -5 else "NOT MET")}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": res, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
