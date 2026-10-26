"""GPTQ-MERGE-BEHAVIOR (docs/REGISTRY.md): GSM8K behavior of error-compensated merges onto frozen NVFP4 / NF4 grids
(bench/gptq_merge_ckpt.py; each evaluated as bf16_base of its checkpoint in results/std_bench/<adapter>_gptq_<fmt>),
against each adapter's existing references: R = (acc - acc_<fmt>_base) / (acc_bf16_exact - acc_bf16_base) (ratio of
counts compared exactly), paired bootstrap 10,000; McNemar against the nearest merge (nvfp4_merge; nf4_merge is PEFT's),
the exact adapter and, for the SFT adapter, the stochastic NVFP4 merge.
  (i)   NVFP4: R >= 0.5 and above the nearest merge (p <= 0.05) on at least 4 of the 6 GRPO adapters
  (ii)  NF4: R >= 0.5 and above PEFT's merge (p <= 0.05) on at least 4 of 6
  (iii) SFT math adapter, NVFP4: R >= 0.8 and above the stochastic merge (p <= 0.05)
    python bench/score_gptq_merge.py [--json out.json]"""
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
GRPO = ("mistral_gsm8k_rl", "mistral_gsm8k_rl_s1", "mistral_gsm8k_rl_s2", "mistral_gsm8k_rlb", "smol_gsm8k_rlb_s0",
        "smol_gsm8k_rlb_s1")
SFT = ("mistral_math",)


def load(f: Path) -> dict | None:
    if not f.exists():
        return None
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def score(name: str) -> dict | None:
    d = RES / name
    ref = {k: load(d / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact")}
    out = {}
    for fmt in ("nvfp4", "nf4"):
        v = {"base": load(d / f"{fmt}_base/gsm8k.jsonl"), "exact": load(d / f"{fmt}_exact/gsm8k.jsonl"),
             "merge": load(d / f"{fmt}_merge/gsm8k.jsonl"), "gptq": load(RES / f"{name}_gptq_{fmt}/bf16_base/gsm8k.jsonl")}
        if fmt == "nvfp4":
            v["sr"] = load(d / "nvfp4_srmerge/gsm8k.jsonl")
        if any(x is None for k, x in v.items() if k != "sr") or any(x is None for x in ref.values()):
            return None
        ids = sorted(set(ref["bf16_base"]) & set(ref["bf16_exact"]) & set(v["base"]) & set(v["gptq"]) & set(v["merge"])
                     & set(v["exact"]))
        c = {k: sum(x[i] for i in ids) for k, x in {**ref, **{k: x for k, x in v.items() if x is not None}}.items()}
        den = c["bf16_exact"] - c["bf16_base"]
        o = {"n": len(ids), "acc": {k: 100 * c[k] / len(ids) for k in c}}
        for arm in ("gptq", "merge", "exact") + (("sr",) if v.get("sr") else ()):
            r = rr.retention(ref["bf16_base"], ref["bf16_exact"], v["base"], v[arm], n_boot=10000)
            o[arm] = {"R": r["R"], "R_ci95": r["R_ci95"], "R_frac": Fraction(c[arm] - c["base"], den) if den else None}
        for other in ("merge", "exact") + (("sr",) if v.get("sr") else ()):
            w, l, p = rr.mcnemar([v["gptq"][i] for i in ids], [v[other][i] for i in ids])
            o[f"gptq_vs_{other}"] = {"gptq_wins": w, "other_wins": l, "p": p}
        out[fmt] = o
    return out


def better(o: dict, other: str) -> bool:
    t = o[f"gptq_vs_{other}"]
    return t["p"] <= 0.05 and t["gptq_wins"] > t["other_wins"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {n: score(n) for n in GRPO + SFT}
    for n, r in res.items():
        if r is None:
            print(f"{n}: incomplete")
            continue
        for fmt, o in r.items():
            acc = o["acc"]
            f = lambda k: f"{acc[k]:.1f} (R {o[k]['R']:.2f} [{o[k]['R_ci95'][0]:.2f}, {o[k]['R_ci95'][1]:.2f}])"  # noqa: E731
            tests = "  ".join(f"vs {k}: {o['gptq_vs_' + k]['gptq_wins']}:{o['gptq_vs_' + k]['other_wins']} "
                              f"p={o['gptq_vs_' + k]['p']:.2g}" for k in ("merge", "exact", "sr") if f"gptq_vs_{k}" in o)
            print(f"{n:20s} {fmt:5s} bf16 {acc['bf16_base']:.1f}->{acc['bf16_exact']:.1f} base {acc['base']:.1f}  "
                  f"GPTQ {f('gptq')}  nearest {f('merge')}  exact {f('exact')}" + (f"  SR {f('sr')}" if "sr" in o else "")
                  + f"\n    {tests}")
    v = {}
    if all(res[n] for n in GRPO):
        half = Fraction(1, 2)
        k1 = sum(res[n]["nvfp4"]["gptq"]["R_frac"] >= half and better(res[n]["nvfp4"], "merge") for n in GRPO)
        k2 = sum(res[n]["nf4"]["gptq"]["R_frac"] >= half and better(res[n]["nf4"], "merge") for n in GRPO)
        v["i"] = f"{'MET' if k1 >= 4 else 'NOT MET'} ({k1} of 6)"
        v["ii"] = f"{'MET' if k2 >= 4 else 'NOT MET'} ({k2} of 6)"
    if res["mistral_math"]:
        o = res["mistral_math"]["nvfp4"]
        v["iii"] = "MET" if o["gptq"]["R_frac"] >= Fraction(4, 5) and better(o, "sr") else "NOT MET"
    print("verdicts:", v)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
