"""QLORA-MERGE (docs/REGISTRY.md, registered 2026-10-14 10:05 UTC, before any data): NF4 variants of the STD-BENCH(-RL)
adapters, R against each results directory's bf16 references and its NF4 base (paired bootstrap, 10,000 resamples).
  (i)   GRPO adapters: PEFT's merge (nf4_merge) keeps R <= 0.5 and less than the exact adapter (McNemar p <= 0.05) on at
        least 3 of 4
  (ii)  GRPO adapters: the exact adapter on the NF4 base keeps R >= 0.5 on at least 3 of 4
  (iii) GRPO adapters: the fresh NF4 quantization of the bf16 merge (nf4_freshmerge) keeps R >= 0.5 on at least 3 of 4
  (iv)  SFT adapters: PEFT's merge keeps R >= 0.8 (math adapter on GSM8K, code adapter on MBPP)
    python bench/score_qlora.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

RES = LB / "results/std_bench"
RL = [("mistral_gsm8k_rl", "gsm8k"), ("mistral_gsm8k_rl_s1", "gsm8k"), ("mistral_gsm8k_rl_s2", "gsm8k"),
      ("mistral_gsm8k_rlb", "gsm8k")]
SFT = [("mistral_math", "gsm8k"), ("mistral_code", "mbpp")]


def load(d: str, v: str, b: str) -> dict | None:
    f = RES / d / v / f"{b}.jsonl"
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def score(d: str, b: str) -> dict | None:
    v = {k: load(d, k, b) for k in ("bf16_base", "bf16_exact", "nf4_base", "nf4_exact", "nf4_merge", "nf4_freshmerge")}
    if any(x is None for x in v.values()):
        return None
    acc = {k: 100 * sum(x.values()) / len(x) for k, x in v.items()}
    out = {"acc": acc}
    for arm in ("nf4_exact", "nf4_merge", "nf4_freshmerge"):
        r = rr.retention(v["bf16_base"], v["bf16_exact"], v["nf4_base"], v[arm], n_boot=10000)
        # the verdicts compare R exactly, from counts: 64/80 must read as 0.8, not 0.7999999999999999 (MBPP, 10-14)
        ids = sorted(set(v["bf16_base"]) & set(v["bf16_exact"]) & set(v["nf4_base"]) & set(v[arm]))
        c = [sum(v[k][i] for i in ids) for k in ("bf16_base", "bf16_exact", "nf4_base", arm)]
        out[arm] = {"R": r["R"], "R_ci95": r["R_ci95"], "R_exact": [c[3] - c[2], c[1] - c[0]],
                    "R_frac": Fraction(c[3] - c[2], c[1] - c[0])}
    ids = sorted(set(v["nf4_exact"]) & set(v["nf4_merge"]))
    w, l, p = rr.mcnemar([v["nf4_exact"][i] for i in ids], [v["nf4_merge"][i] for i in ids])
    out["exact_vs_merge"] = {"exact_wins": w, "merge_wins": l, "p": p}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for d, b in RL + SFT:
        r = score(d, b)
        res[d] = r
        if r is None:
            print(f"{d}: incomplete")
            continue
        acc = r["acc"]
        print(f"{d:22s} {b:6s} bf16 {acc['bf16_base']:.1f}->{acc['bf16_exact']:.1f}  NF4 base {acc['nf4_base']:.1f}  "
              + "  ".join(f"{k[4:]} {acc[k]:.1f} (R {r[k]['R']:.2f} [{r[k]['R_ci95'][0]:.2f}, {r[k]['R_ci95'][1]:.2f}])"
                          for k in ("nf4_exact", "nf4_merge", "nf4_freshmerge"))
              + f"  exact vs merge {r['exact_vs_merge']['exact_wins']}:{r['exact_vs_merge']['merge_wins']} p={r['exact_vs_merge']['p']:.2g}")
    rl = [res[d] for d, _ in RL if res.get(d)]
    sft = [res[d] for d, _ in SFT if res.get(d)]
    v = {}
    if len(rl) == 4:
        v["i"] = sum(r["nf4_merge"]["R_frac"] <= Fraction(1, 2) and r["exact_vs_merge"]["p"] <= 0.05
                     and r["exact_vs_merge"]["exact_wins"] > r["exact_vs_merge"]["merge_wins"] for r in rl) >= 3
        v["ii"] = sum(r["nf4_exact"]["R_frac"] >= Fraction(1, 2) for r in rl) >= 3
        v["iii"] = sum(r["nf4_freshmerge"]["R_frac"] >= Fraction(1, 2) for r in rl) >= 3
    if len(sft) == 2:
        v["iv"] = all(r["nf4_merge"]["R_frac"] >= Fraction(4, 5) for r in sft)
    print("verdicts:", {k: ("MET" if x else "NOT MET") for k, x in v.items()})
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
