"""GPTQ-FP8 (docs/REGISTRY.md): GPTQ merges onto the frozen per-row FP8 grid, GSM8K. R = (acc - acc_fp8_base) /
(acc_bf16_exact - acc_bf16_base) (paired bootstrap 10,000); McNemar against the nearest FP8 merge (fp8_merge).
  (i) SmolLM2 smol_gsm8k_rlb_s0..s3: R >= 0.7 and above the nearest merge (p <= 0.05) on at least 3 of 4.
    python bench/score_gptq_fp8.py [--json out.json]"""
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
SMOL = tuple(f"smol_gsm8k_rlb_s{k}" for k in range(4))
MIS = ("mistral_gsm8k_rl", "mistral_gsm8k_rl_s1", "mistral_gsm8k_rl_s2", "mistral_gsm8k_rlb")


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def score(n: str) -> dict | None:
    d = RES / n
    v = {k: load(d / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact", "fp8_base", "fp8_merge", "fp8_exact")}
    v["gptq"] = load(RES / f"{n}_gptq_fp8/bf16_base/gsm8k.jsonl")
    if any(x is None for x in v.values()):
        return None
    ids = sorted(set.intersection(*(set(x) for x in v.values())))
    c = {k: sum(x[i] for i in ids) for k, x in v.items()}
    out = {"acc": {k: 100 * c[k] / len(ids) for k in c}}
    for arm in ("gptq", "fp8_merge", "fp8_exact"):
        r = rr.retention(v["bf16_base"], v["bf16_exact"], v["fp8_base"], v[arm], n_boot=10000)
        out[arm] = {"R": r["R"], "R_ci95": r["R_ci95"], "R_frac": Fraction(c[arm] - c["fp8_base"], c["bf16_exact"] - c["bf16_base"])}
    w, l, p = rr.mcnemar([v["gptq"][i] for i in ids], [v["fp8_merge"][i] for i in ids])
    out["gptq_vs_merge"] = {"wins": w, "losses": l, "p": p}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {n: score(n) for n in SMOL + MIS}
    for n, r in res.items():
        if r is None:
            print(n, "incomplete")
            continue
        t = r["gptq_vs_merge"]
        print(f"{n:22s} fp8 base {r['acc']['fp8_base']:.1f}  GPTQ {r['acc']['gptq']:.1f} (R {r['gptq']['R']:.2f} "
              f"[{r['gptq']['R_ci95'][0]:.2f}, {r['gptq']['R_ci95'][1]:.2f}])  nearest {r['acc']['fp8_merge']:.1f} "
              f"(R {r['fp8_merge']['R']:.2f})  exact R {r['fp8_exact']['R']:.2f}  GPTQ vs nearest {t['wins']}:{t['losses']} p={t['p']:.2g}")
    v = {}
    if all(res[n] for n in SMOL):
        k = sum(res[n]["gptq"]["R_frac"] >= Fraction(7, 10) and res[n]["gptq_vs_merge"]["p"] <= 0.05
                and res[n]["gptq_vs_merge"]["wins"] > res[n]["gptq_vs_merge"]["losses"] for n in SMOL)
        v["i"] = f"{'MET' if k >= 3 else 'NOT MET'} ({k} of 4)"
    print("verdicts:", v)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
