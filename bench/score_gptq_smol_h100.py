"""GPTQ-SMOL-H100 (docs/REGISTRY.md): SmolLM2-1.7B GRPO seeds 4 and 5 (smol_gsm8k_rlb_s4/_s5), every variant and the
GPTQ merges onto the frozen NVFP4, NF4 and FP8 grids evaluated on one H100 per seed. R = (acc - acc_<fmt>_base) /
(acc_bf16_exact - acc_bf16_base) (paired bootstrap 10,000); judged if the bf16 gain is at least 3 points.
  (i)  FP8: the GPTQ merge keeps R >= 0.7 and more than the nearest FP8 merge (McNemar p <= 0.05), each judged seed.
  (ii) NVFP4: the GPTQ merge keeps more than the nearest NVFP4 merge (p <= 0.05), each judged seed.
Descriptive: NF4 (GPTQ against PEFT's merge) and STD-BENCH-RL-S's (i)-(iii) on these seeds.
    python3 bench/score_gptq_smol_h100.py [--seeds 4 5] [--json results/std_bench/gptq_smol_h100_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

VARIANTS = ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_merge", "nf4_base", "nf4_exact", "nf4_merge",
            "fp8_base", "fp8_exact", "fp8_merge")


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def acc(x: dict) -> float:
    return 100 * sum(x.values()) / len(x)


def mc(a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    w, l, p = rr.mcnemar([a[i] for i in ids], [b[i] for i in ids])
    return {"wins": w, "losses": l, "p": p}


def score(seed: int) -> dict | None:
    n = f"smol_gsm8k_rlb_s{seed}"
    v = {k: load(RES / n / k / "gsm8k.jsonl") for k in VARIANTS}
    for fmt in ("nvfp4", "nf4", "fp8"):
        v[f"{fmt}_gptq"] = load(RES / f"{n}_gptq_{fmt}/bf16_base/gsm8k.jsonl")
    if any(x is None for x in v.values()):
        return None
    gain = acc(v["bf16_exact"]) - acc(v["bf16_base"])
    out = {"gain_pts": gain, "judged": gain >= 3, "acc": {k: acc(x) for k, x in v.items()}}
    for fmt in ("nvfp4", "nf4", "fp8"):
        for kind in ("merge", "exact", "gptq"):
            r = rr.retention(v["bf16_base"], v["bf16_exact"], v[f"{fmt}_base"], v[f"{fmt}_{kind}"], n_boot=10000)
            out[f"{fmt}_{kind}"] = {"R": r["R"], "R_ci95": r["R_ci95"]}
        out[f"{fmt}_gptq_vs_merge"] = mc(v[f"{fmt}_gptq"], v[f"{fmt}_merge"])
        out[f"{fmt}_exact_vs_merge"] = mc(v[f"{fmt}_exact"], v[f"{fmt}_merge"])
    return out


def above(r: dict, fmt: str, key: str = "gptq_vs_merge") -> bool:
    m = r[f"{fmt}_{key}"]
    return m["wins"] > m["losses"] and m["p"] <= 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[4, 5])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = {str(s): score(s) for s in a.seeds}
    judged = [r for r in rows.values() if r and r["judged"]]
    for s, r in rows.items():
        if r is None:
            print(f"seed {s}: not complete")
            continue
        f2 = lambda k: f"{r[k]['R']:.2f} [{r[k]['R_ci95'][0]:.2f}, {r[k]['R_ci95'][1]:.2f}]"  # noqa: E731
        print(f"seed {s}: bf16 {r['acc']['bf16_base']:.1f} -> {r['acc']['bf16_exact']:.1f} ({r['gain_pts']:+.1f}, "
              f"{'judged' if r['judged'] else 'not judged'})")
        for fmt in ("fp8", "nvfp4", "nf4"):
            g, e = r[f"{fmt}_gptq_vs_merge"], r[f"{fmt}_exact_vs_merge"]
            print(f"  {fmt}: GPTQ {f2(fmt + '_gptq')}, nearest/PEFT merge {f2(fmt + '_merge')}, exact {f2(fmt + '_exact')}; "
                  f"GPTQ vs merge {g['wins']}:{g['losses']} p {g['p']:.2g}; exact vs merge {e['wins']}:{e['losses']} p {e['p']:.2g}")
    verdict = {}
    if judged:
        verdict["i"] = "MET" if all(r["fp8_gptq"]["R"] >= 0.7 and above(r, "fp8") for r in judged) else "NOT MET"
        verdict["ii"] = "MET" if all(above(r, "nvfp4") for r in judged) else "NOT MET"
        verdict["rls_i"] = "MET" if all(r["nvfp4_merge"]["R"] <= 0.5 and above(r, "nvfp4", "exact_vs_merge")
                                        for r in judged) else "NOT MET"
        verdict["rls_ii"] = "MET" if all(r["nf4_merge"]["R"] <= 0.5 and above(r, "nf4", "exact_vs_merge")
                                         for r in judged) else "NOT MET"
        verdict["rls_iii"] = "MET" if all(r["fp8_exact"]["R"] >= 0.8 for r in judged) else "NOT MET"
        verdict = {k: f"{v} ({len(judged)} judged)" for k, v in verdict.items()}
    else:
        verdict = {"all": "not judged" if all(r is not None for r in rows.values()) else "incomplete"}
    print("verdict:", verdict)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
