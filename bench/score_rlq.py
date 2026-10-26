"""STD-BENCH-RL-Q (docs/REGISTRY.md, registered 2026-10-14 11:13 UTC, before any data): Qwen2.5-1.5B-Instruct GRPO
adapters on GSM8K (seeds 0/1 at lr 1e-5 and 5e-5). R of each variant against the bf16 references and its format's own
base (paired bootstrap 10,000; the verdicts compare the ratio of counts exactly). Judged if the bf16 gain >= 3 points.
  (i)   nearest NVFP4 merge: R <= 0.5 and below the exact adapter on NVFP4 (McNemar p <= 0.05)
  (ii)  PEFT's NF4 merge: R <= 0.5 and below the exact adapter on NF4 (p <= 0.05)
  (iii) exact adapter on FP8: R >= 0.8
  each met if it holds on >= min(3, n) of the n judged adapters; not judged if n < 2.
--family smol: STD-BENCH-RL-S (registered 2026-10-14 14:04 UTC), the same rule on SmolLM2-1.7B-Instruct adapters.
    python bench/score_rlq.py [--family qwen15|smol] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

RES = LB / "results/std_bench"
FAMILY = "qwen15"
ADAPTERS = tuple(f"{FAMILY}_gsm8k_{k}" for k in ("rl_s0", "rl_s1", "rlb_s0", "rlb_s1"))
FMTS = {"nvfp4": ("nvfp4_exact", "nvfp4_merge"), "nf4": ("nf4_exact", "nf4_merge"), "fp8": ("fp8_exact", "fp8_merge")}


def load(d: Path, v: str) -> dict | None:
    f = d / v / "gsm8k.jsonl"
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def weight_gain(name: str, variant: str) -> float | None:
    f = LB / "logs" / (f"lb_rls_eval_{name}.out" if name.startswith("smol") else f"lb_rlq_eval_{name}.out")
    hits = re.findall(rf'^{variant} applied: .*?"gain": ([-0-9.e]+)', f.read_text(), re.M) if f.exists() else []
    return float(hits[-1]) if hits else None


def score(name: str) -> dict | None:
    d = RES / name
    names = ["bf16_base", "bf16_exact"] + [x for f in FMTS for x in (f"{f}_base",) + FMTS[f]]
    v = {k: load(d, k) for k in names}
    if any(x is None for x in v.values()):
        return None
    ids = sorted(set.intersection(*(set(x) for x in v.values())))
    c = {k: sum(v[k][i] for i in ids) for k in names}
    n = len(ids)
    out = {"n": n, "acc": {k: 100 * c[k] / n for k in names}, "gain_pts": 100 * (c["bf16_exact"] - c["bf16_base"]) / n}
    out["judged"] = out["gain_pts"] >= 3
    den = c["bf16_exact"] - c["bf16_base"]
    for f, arms in FMTS.items():
        for arm in arms:
            r = rr.retention(v["bf16_base"], v["bf16_exact"], v[f"{f}_base"], v[arm], n_boot=10000)
            out[arm] = {"R": r["R"], "R_ci95": r["R_ci95"], "R_exact": [c[arm] - c[f"{f}_base"], den],
                        "R_frac": Fraction(c[arm] - c[f"{f}_base"], den) if den else None,
                        "weight_gain": weight_gain(name, arm) if arm.endswith("merge") else None}
        w, l, p = rr.mcnemar([v[arms[0]][i] for i in ids], [v[arms[1]][i] for i in ids])
        out[f"{f}_exact_vs_merge"] = {"exact_wins": w, "merge_wins": l, "p": p}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--min-judged", type=int, default=2, help="the replications registered 16:12 judge with 1")
    ap.add_argument("--family", default="qwen15", choices=("qwen15", "smol"))
    ap.add_argument("--adapters", nargs="+", default=None, help="STD-BENCH-RL-S2: smol_gsm8k_rlb_s2 smol_gsm8k_rlb_s3")
    a = ap.parse_args()
    global ADAPTERS
    ADAPTERS = tuple(a.adapters) if a.adapters else tuple(f"{a.family}_gsm8k_{k}" for k in ("rl_s0", "rl_s1", "rlb_s0", "rlb_s1"))
    res = {n: score(n) for n in ADAPTERS}
    for name, r in res.items():
        if r is None:
            print(f"{name}: incomplete")
            continue
        acc = r["acc"]
        cells = "  ".join(f"{k} {acc[k]:.1f} (R {r[k]['R']:.2f} [{r[k]['R_ci95'][0]:.2f}, {r[k]['R_ci95'][1]:.2f}])"
                          for f in FMTS for k in FMTS[f])
        tests = "  ".join(f"{f}: {r[f + '_exact_vs_merge']['exact_wins']}:{r[f + '_exact_vs_merge']['merge_wins']} "
                          f"p={r[f + '_exact_vs_merge']['p']:.2g}" for f in FMTS)
        print(f"{name}: bf16 {acc['bf16_base']:.1f}->{acc['bf16_exact']:.1f} ({r['gain_pts']:+.1f}, "
              f"{'judged' if r['judged'] else 'NOT judged'}); bases nvfp4 {acc['nvfp4_base']:.1f} nf4 {acc['nf4_base']:.1f} "
              f"fp8 {acc['fp8_base']:.1f}\n    {cells}\n    exact vs merge {tests}; g_W merges "
              f"{[r[k]['weight_gain'] for k in ('nvfp4_merge', 'nf4_merge', 'fp8_merge')]}")
    done = [r for r in res.values() if r is not None]
    judged = [r for r in done if r["judged"]]
    v = {}
    if len(done) == len(ADAPTERS):
        n = len(judged)
        if n < a.min_judged:
            v = {k: "not judged" for k in ("i", "ii", "iii")}
        else:
            need = min(3, n)

            def erased(r, f):
                t = r[f"{f}_exact_vs_merge"]
                return (r[f"{f}_merge"]["R_frac"] <= Fraction(1, 2) and t["p"] <= 0.05
                        and t["exact_wins"] > t["merge_wins"])
            k = {"i": sum(erased(r, "nvfp4") for r in judged), "ii": sum(erased(r, "nf4") for r in judged),
                 "iii": sum(r["fp8_exact"]["R_frac"] >= Fraction(4, 5) for r in judged)}
            v = {key: f"{'MET' if x >= need else 'NOT MET'} ({x} of {n} judged)" for key, x in k.items()}
    print("verdicts:", v or "waiting for all adapters")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
