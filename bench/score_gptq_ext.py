"""GPTQ-CALIB and GPTQ-QLORA (docs/REGISTRY.md, registered 2026-10-14 before any data).
GPTQ-CALIB: the four Mistral GRPO adapters' GPTQ merges onto the frozen NVFP4 grid calibrated on Magicoder-OSS code
(results/std_bench/<adapter>_gptqcal_nvfp4) against the nearest merge and the GSM8K-calibrated GPTQ merge.
  (i) R above the nearest merge (McNemar p <= 0.05) on at least 3 of 4; descriptive: difference to the in-domain merge.
GPTQ-QLORA: the four QLoRA-trained Mistral GRPO adapters' GPTQ merges onto the frozen NF4 grid
(results/std_bench/mistral_gsm8k_qlora_s<k>_gptq_nf4), R_q against the trained policy (NF4 base + exact adapter).
  (i) R_q above PEFT's merge (McNemar p <= 0.05) on at least 3 of 4; descriptive: R_q.
    python bench/score_gptq_ext.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

RES = LB / "results/std_bench"
GRPO = ("mistral_gsm8k_rl", "mistral_gsm8k_rl_s1", "mistral_gsm8k_rl_s2", "mistral_gsm8k_rlb")


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def compare(ref_base, ref_top, base_q, arms: dict, against: str) -> dict | None:
    if any(x is None for x in (ref_base, ref_top, base_q, *arms.values())):
        return None
    ids = sorted(set(ref_base) & set(ref_top) & set(base_q) & set.intersection(*(set(x) for x in arms.values())))
    out = {}
    for k, v in arms.items():
        r = rr.retention(ref_base, ref_top, base_q, v, n_boot=10000)
        out[k] = {"acc": 100 * sum(v[i] for i in ids) / len(ids), "R": r["R"], "R_ci95": r["R_ci95"]}
    for k in arms:
        if k != against:
            w, l, p = rr.mcnemar([arms[k][i] for i in ids], [arms[against][i] for i in ids])
            out[f"{k}_vs_{against}"] = {"wins": w, "losses": l, "p": p}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {"calib": {}, "qlora": {}}
    for n in GRPO:
        d = RES / n
        res["calib"][n] = compare(load(d / "bf16_base/gsm8k.jsonl"), load(d / "bf16_exact/gsm8k.jsonl"),
                                  load(d / "nvfp4_base/gsm8k.jsonl"),
                                  {"gptq_code": load(RES / f"{n}_gptqcal_nvfp4/bf16_base/gsm8k.jsonl"),
                                   "gptq_gsm8k": load(RES / f"{n}_gptq_nvfp4/bf16_base/gsm8k.jsonl"),
                                   "nearest": load(d / "nvfp4_merge/gsm8k.jsonl")}, "nearest")
    for s in range(4):
        d = RES / f"mistral_gsm8k_qlora_s{s}"
        res["qlora"][s] = compare(load(d / "nf4_base/gsm8k.jsonl"), load(d / "nf4_exact/gsm8k.jsonl"),
                                  load(d / "nf4_base/gsm8k.jsonl"),
                                  {"gptq": load(RES / f"mistral_gsm8k_qlora_s{s}_gptq_nf4/bf16_base/gsm8k.jsonl"),
                                   "peft": load(d / "nf4_merge/gsm8k.jsonl")}, "peft")
    for part, rows in res.items():
        for k, r in rows.items():
            print(part, k, "incomplete" if r is None else json.dumps(
                {x: (round(y["R"], 3), round(y["acc"], 1)) if "R" in y else (y["wins"], y["losses"], f"{y['p']:.2g}")
                 for x, y in r.items()}))
    v = {}
    c = [r for r in res["calib"].values() if r]
    if len(c) == 4:
        k = sum(r["gptq_code_vs_nearest"]["p"] <= 0.05 and r["gptq_code_vs_nearest"]["wins"] > r["gptq_code_vs_nearest"]["losses"]
                for r in c)
        v["calib_i"] = f"{'MET' if k >= 3 else 'NOT MET'} ({k} of 4)"
    q = [r for r in res["qlora"].values() if r]
    if len(q) == 4:
        k = sum(r["gptq_vs_peft"]["p"] <= 0.05 and r["gptq_vs_peft"]["wins"] > r["gptq_vs_peft"]["losses"] for r in q)
        v["qlora_i"] = f"{'MET' if k >= 3 else 'NOT MET'} ({k} of 4)"
    print("verdicts:", v)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
