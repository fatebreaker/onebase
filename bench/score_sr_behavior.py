"""SR-BEHAVIOR (docs/REGISTRY.md, registered 2026-10-13 19:32 UTC, before any data): retention R of the stochastic NVFP4
merge (rl_nvfp4srfrozen) against P0-ACCURACY (a)/(a2)'s references in results/p0_accuracy/behavior/<S1|S2>, verified
accuracy, paired bootstrap 95% CI with 5,000 resamples (as (a)); MET if R >= 0.8. Secondary: McNemar against the exact
adapter (rl_nvfp4unmerged) and the nearest merge (rl_nvfp4frozen).
    python bench/score_sr_behavior.py [S1 S2] [--json out.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

BENCH = {"S1": "medcalc_ver_s220", "S2": "ehrsql2024_s300", "REL": "medcalc_ver_s220"}
ML = Path("$ROUNDING")
# REL = SR-BEHAVIOR-REL (registered 20:35 UTC): NVIDIA's NVFP4 release, W1-NVFP4-BEHAVIOR's references (10,000 resamples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", default=["S1", "S2"])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for subj in a.subjects:
        b = BENCH[subj]
        if subj == "REL":
            d, nb = ML / "results/w1_nvfp4_behavior", 10000
            roles = {"base": "bf16_base", "ref": "bf16_exact", "base_q": "nvfp4_base", "sr": "nvfp4_srfrozen",
                     "exact": "nvfp4_exact", "merge": "nvfp4_frozen"}
        else:
            d, nb = LB / "results/p0_accuracy/behavior" / subj, 5000
            roles = {"base": "ref_bf16base", "ref": "ref_bf16lora", "base_q": "base_nvfp4", "sr": "rl_nvfp4srfrozen",
                     "exact": "rl_nvfp4unmerged", "merge": "rl_nvfp4frozen"}
        acc = {}
        for k, tag in roles.items():
            p = d / tag / b
            acc[k] = rr.load(str(p), b, "verified") if (p / "summary.json").exists() else None
        if any(acc[k] is None for k in ("base", "ref", "base_q", "sr")):
            print(f"{subj}: incomplete ({[k for k, v in acc.items() if v is None]})")
            continue
        r = rr.retention(acc["base"], acc["ref"], acc["base_q"], acc["sr"], n_boot=nb)
        res = {"R": r["R"], "R_ci95": r["R_ci95"], "met": r["R"] >= 0.8,
               "acc_sr": 100 * sum(acc["sr"].values()) / len(acc["sr"])}
        for k in ("exact", "merge"):
            if acc[k] is not None:
                ids = sorted(set(acc["sr"]) & set(acc[k]))
                w, l, p = rr.mcnemar([acc["sr"][i] for i in ids], [acc[k][i] for i in ids])
                res[f"vs_{k}"] = {"sr_wins": w, f"{k}_wins": l, "p": p,
                                  f"acc_{k}": 100 * sum(acc[k].values()) / len(acc[k])}
        out[subj] = res
        print(f"{subj}: R = {r['R']:.2f} [{r['R_ci95'][0]:.2f}, {r['R_ci95'][1]:.2f}] -> {'MET' if res['met'] else 'NOT MET'}"
              f"; accuracy {res['acc_sr']:.1f}%; " + "; ".join(
                  f"vs {k}: {v['sr_wins']} vs {v[k + '_wins']} discordant, p = {v['p']:.3g} ({k} {v['acc_' + k]:.1f}%)"
                  for k, v in ((k, res.get('vs_' + k)) for k in ('exact', 'merge')) if v))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
