"""GPTQ-REL (docs/REGISTRY.md): the Llama-3.1-8B MedCalc RL agent merged onto NVIDIA's NVFP4 release with the
error-compensated merge (official_nvfp4.py build --merge gptq, calibrated on GSM8K or on Magicoder-OSS code), evaluated
as W1-NVFP4-BEHAVIOR's variants (verified accuracy, medcalc_ver_s220; mlsys results/w1_nvfp4_behavior/<variant>).
R = (acc - acc_nvfp4_base) / (acc_bf16_exact - acc_bf16_base), paired bootstrap 10,000; McNemar against the nearest
merge on the release's scales (nvfp4_frozen), the exact adapter and the stochastic merge (nvfp4_srfrozen).
  (i)  GSM8K-calibrated: R >= 0.5 and above the nearest merge (p <= 0.05)
  (ii) code-calibrated: R >= 0.5 and above the nearest merge (p <= 0.05)
GPTQ-REL-SFT: --root results/w1_nvfp4_behavior_sft (MedCalc SFT agent) or _sftehr --bench ehrsql2024_s300 (EHR SQL
SFT agent), GSM8K calibration only; the same rule (i).
    python bench/score_gptq_rel.py [--root DIR --bench B] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML = Path("$ROUNDING")
sys.path.insert(0, str(ML / "analysis"))
import read_retention as rr  # noqa: E402

ROOT = ML / "results/w1_nvfp4_behavior"
B = "medcalc_ver_s220"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--bench", default=B)
    a = ap.parse_args()
    root, bench = Path(a.root), a.bench
    v = {}
    for name in ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_frozen", "nvfp4_srfrozen",
                 "nvfp4_gptq_gsm8k", "nvfp4_gptq_code", "nvfp4_gptq_task", "nvfp4_gptq_trans"):
        d = root / name / bench
        v[name] = rr.load(str(d), bench, "verified") if (d / "summary.json").exists() else None
    res, verdict = {}, {}
    for key, arm in (("i", "nvfp4_gptq_gsm8k"), ("ii", "nvfp4_gptq_code"), ("task", "nvfp4_gptq_task"),
                     ("trans", "nvfp4_gptq_trans")):
        if v[arm] is None:
            print(arm, "not evaluated yet")
            continue
        r = rr.retention(v["bf16_base"], v["bf16_exact"], v["nvfp4_base"], v[arm], n_boot=10000)
        ids = sorted(set(v[arm]) & set(v["nvfp4_frozen"]) & set(v["nvfp4_exact"]))
        o = {"acc": 100 * sum(v[arm][i] for i in ids) / len(ids), "R": r["R"], "R_ci95": r["R_ci95"]}
        for other in ("nvfp4_frozen", "nvfp4_exact", "nvfp4_srfrozen"):
            if v.get(other) is not None:
                w, l, p = rr.mcnemar([v[arm][i] for i in ids], [v[other][i] for i in ids])
                o[f"vs_{other}"] = {"wins": w, "losses": l, "p": p}
        res[arm] = o
        t = o["vs_nvfp4_frozen"]
        verdict[key] = "MET" if r["R"] >= 0.5 and t["p"] <= 0.05 and t["wins"] > t["losses"] else "NOT MET"
        print(arm, json.dumps(o))
    print("verdicts:", verdict)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": res, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
