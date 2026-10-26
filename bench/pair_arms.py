"""Descriptive paired comparison of two (a)/(a2) arms on the same problems (verified accuracy, McNemar), per subject:
the exact adapter vs the NVFP4-stored adapter, both unmerged on the NVFP4 base.
    python bench/pair_arms.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402

B = Path(__file__).resolve().parents[1] / "results/p0_accuracy/behavior"
for subj, bench in (("S1", "medcalc_ver_s220"), ("S2", "ehrsql2024_s300")):
    a = rr.load(str(B / subj / "rl_nvfp4unmerged" / bench), bench, "verified")
    b = rr.load(str(B / subj / "rl_nvfp4unmerged_l4" / bench), bench, "verified")
    ids = sorted(set(a) & set(b))
    xa, xb = [a[i] for i in ids], [b[i] for i in ids]
    w_a, w_b, p = rr.mcnemar(xa, xb)
    print(f"{subj}: exact {100 * sum(xa) / len(ids):.1f}%  stored {100 * sum(xb) / len(ids):.1f}%  "
          f"(exact wins {w_a}, stored wins {w_b}, McNemar p = {p:.3g}, n = {len(ids)})")
