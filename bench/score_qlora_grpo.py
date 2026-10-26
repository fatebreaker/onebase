"""QLORA-GRPO (docs/REGISTRY.md, registered 2026-10-14 10:31 UTC, before any data): GRPO adapters trained QLoRA-style on
the NF4 base (seeds 0-3), on GSM8K test. R_q = (acc_variant - acc_nf4_base) / (acc_nf4_exact - acc_nf4_base), the share
of the trained policy's gain a variant keeps, compared exactly from counts; paired bootstrap (10,000) and McNemar against
nf4_exact. Gate: a seed is judged only if nf4_exact - nf4_base >= 5 points.
  (i)  PEFT's merge (nf4_merge) keeps R_q <= 0.5 and is below the exact adapter (McNemar p <= 0.05)
  (ii) the fresh NF4 quantization of the bf16 merge (nf4_freshmerge) keeps R_q >= 0.5
  each met if it holds on at least min(3, n) of the n judged seeds; not judged if n < 2.
Descriptive: the PEFT merge's weight-level gain (from the eval log) and the adapter on the bf16 master (bf16_exact vs
bf16_base).
    python bench/score_qlora_grpo.py [--json out.json]"""
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
SEEDS = (0, 1, 2, 3)
FAMILY = "mistral"                       # --family smol: QLORA-GRPO-S (seeds 0 and 1, adapters smol_gsm8k_qlora_s*)
VARS = ("nf4_base", "nf4_exact", "nf4_merge", "nf4_freshmerge", "bf16_base", "bf16_exact")


def load(d: Path, v: str) -> dict | None:
    f = d / v / "gsm8k.jsonl"
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def weight_gain(seed: int, variant: str) -> float | None:
    f = LB / "logs" / (f"lb_qgrpo_eval_s{seed}.out" if FAMILY == "mistral" else f"lb_qgrpos_eval_s{seed}.out")
    hits = re.findall(rf'^{variant} applied: .*?"gain": ([-0-9.e]+)', f.read_text(), re.M) if f.exists() else []
    return float(hits[-1]) if hits else None


def score(seed: int) -> dict | None:
    d = RES / f"{FAMILY}_gsm8k_qlora_s{seed}"
    v = {k: load(d, k) for k in VARS}
    if any(x is None for x in v.values()):
        return None
    ids = sorted(set.intersection(*(set(x) for x in v.values())))
    c = {k: sum(v[k][i] for i in ids) for k in VARS}
    n = len(ids)
    out = {"n": n, "acc": {k: 100 * c[k] / n for k in VARS}, "gain_pts": 100 * (c["nf4_exact"] - c["nf4_base"]) / n}
    out["judged"] = out["gain_pts"] >= 5
    for arm in ("nf4_merge", "nf4_freshmerge"):
        # R_q: the trained policy (NF4 base + exact adapter) is the reference, the NF4 base its own base
        r = rr.retention(v["nf4_base"], v["nf4_exact"], v["nf4_base"], v[arm], n_boot=10000)
        den = c["nf4_exact"] - c["nf4_base"]
        out[arm] = {"R": r["R"], "R_ci95": r["R_ci95"], "R_exact": [c[arm] - c["nf4_base"], den],
                    "R_frac": Fraction(c[arm] - c["nf4_base"], den) if den else None}
    w, l, p = rr.mcnemar([v["nf4_exact"][i] for i in ids], [v["nf4_merge"][i] for i in ids])
    out["exact_vs_merge"] = {"exact_wins": w, "merge_wins": l, "p": p}
    out["bf16_gain_pts"] = 100 * (c["bf16_exact"] - c["bf16_base"]) / n
    out["merge_weight_gain"] = weight_gain(seed, "nf4_merge")
    out["freshmerge_weight_gain"] = weight_gain(seed, "nf4_freshmerge")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--min-judged", type=int, default=2, help="the replications registered 16:12 judge with 1")
    ap.add_argument("--family", default="mistral", choices=("mistral", "smol"))
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="QLORA-GRPO-S2: --family smol --seeds 2 3")
    a = ap.parse_args()
    global FAMILY, SEEDS
    FAMILY = a.family
    if FAMILY == "smol":
        SEEDS = tuple(a.seeds) if a.seeds else (0, 1)
    res = {s: score(s) for s in SEEDS}
    for s, r in res.items():
        if r is None:
            print(f"seed {s}: incomplete")
            continue
        acc = r["acc"]
        print(f"seed {s}: NF4 base {acc['nf4_base']:.1f} -> trained {acc['nf4_exact']:.1f} ({r['gain_pts']:+.1f}, "
              f"{'judged' if r['judged'] else 'NOT judged'}); PEFT merge {acc['nf4_merge']:.1f} "
              f"(R_q {r['nf4_merge']['R']:.2f} [{r['nf4_merge']['R_ci95'][0]:.2f}, {r['nf4_merge']['R_ci95'][1]:.2f}]); "
              f"fresh NF4 of bf16 merge {acc['nf4_freshmerge']:.1f} (R_q {r['nf4_freshmerge']['R']:.2f} "
              f"[{r['nf4_freshmerge']['R_ci95'][0]:.2f}, {r['nf4_freshmerge']['R_ci95'][1]:.2f}]); exact vs merge "
              f"{r['exact_vs_merge']['exact_wins']}:{r['exact_vs_merge']['merge_wins']} p={r['exact_vs_merge']['p']:.2g}; "
              f"bf16 {acc['bf16_base']:.1f} -> {acc['bf16_exact']:.1f}; g_W merge {r['merge_weight_gain']}")
    done = [r for r in res.values() if r is not None]
    judged = [r for r in done if r["judged"]]
    v = {}
    if len(done) == len(SEEDS):
        n = len(judged)
        if n < a.min_judged:
            v = {"i": "not judged", "ii": "not judged"}
        else:
            need = min(3, n)
            ok_i = sum(r["nf4_merge"]["R_frac"] <= Fraction(1, 2) and r["exact_vs_merge"]["p"] <= 0.05
                       and r["exact_vs_merge"]["exact_wins"] > r["exact_vs_merge"]["merge_wins"] for r in judged)
            ok_ii = sum(r["nf4_freshmerge"]["R_frac"] >= Fraction(1, 2) for r in judged)
            v = {"i": f"{'MET' if ok_i >= need else 'NOT MET'} ({ok_i} of {n} judged)",
                 "ii": f"{'MET' if ok_ii >= need else 'NOT MET'} ({ok_ii} of {n} judged)"}
    print("verdicts:", v or "waiting for all seeds")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
