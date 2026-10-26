"""Score RL-UNCORRECTED-L40S (docs/REGISTRY.md; written before any data): arm S8ULF, S8TLF's recipe without TIS (the W8A8 engine with
the plugin, two copies, the weight-matched W8A16 trainer with flash-linear-attention) on one L40S, seeds 0-2.
  (i)  every S8ULF run's mean trainer-engine mismatch over steps 151-200 is at least 2x the matched engine's (S8ML: W8A16 engine and
       trainer, no correction, the same GPU and recipe; S8ML's runs pooled over the same steps)
Descriptive: each run's late mismatch against S8TLF's (W8A8 engine, W8A16 trainer, TIS) and S8ML's; final accuracy (OlympiadBench +
Minerva pooled, own FP8 engine) minus S0OL's (Welch) and whether any run ends at or below its pre-RL accuracy. Stopping: runs trained
and evaluated by 2026-10-26 12:00 UTC.
    python3 bench/score_rlul.py -> results/rlul_verdict.json"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

for _a in ("S8ULF", "S8UF", "S8TLF", "S8ML", "S8MH", "S8MS9L", "S8WA"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))
Q.EV.setdefault("S0OL", ("bf16_base", "bf16_exact"))


def late(arm: str, s: int) -> float | None:
    rs = Q.rows(arm, s)
    v = [r["mismatch"] for r in rs if r.get("mismatch") is not None and 151 <= r["step"] <= 200]
    return st.fmean(v) if v and rs and rs[-1]["step"] >= 200 else None


def main() -> int:
    out: dict = {}
    lm = {arm: {k: m for k in range(7) if (m := late(arm, k)) is not None} for arm in ("S8ULF", "S8ML", "S8TLF")}
    out["late_mismatch"] = lm
    if lm["S8ULF"] and lm["S8ML"]:
        ref = st.fmean(lm["S8ML"].values())
        out["i"] = {"S8ML_late": ref, "ratios": {k: v / ref for k, v in lm["S8ULF"].items()}, "n": len(lm["S8ULF"]),
                    "verdict": "MET" if all(v >= 2 * ref for v in lm["S8ULF"].values()) else "NOT MET"}
    acc = {}
    for arm in ("S8ULF", "S0OL"):
        acc[arm] = {}
        for k in range(7):
            x = Q.summary(arm, k)
            if x and x.get("hard_final") is not None:
                acc[arm][k] = (x["hard_final"], x.get("hard_base"))
    a = [v[0] for v in acc["S8ULF"].values()]
    b = [v[0] for v in acc["S0OL"].values()]
    out["accuracy"] = {"S8ULF": acc["S8ULF"], "minus_S0OL": welch(a, b) if len(a) >= 2 and len(b) >= 2 else None,
                       "at_or_below_base": [k for k, (f, base) in acc["S8ULF"].items() if base is not None and not f > base]}
    # pooled with RL-UNCORRECTED-FAST: uncorrected W8A8 beside the FLA W8A16 trainer against the unrounded engines' uncorrected runs
    low, n = 0, 0
    for arm, seeds in (("S8ULF", range(3)), ("S8UF", range(6))):
        for k in seeds:
            x = Q.summary(arm, k)
            if x and x.get("hard_final") is not None and x.get("hard_base") is not None:
                n += 1
                low += not x["hard_final"] > x["hard_base"]
    low_u, n_u = 0, 0
    for arm, seeds in (("S8MH", range(6)), ("S8ML", range(3)), ("S8MS9L", range(5)), ("S8WA", range(4))):
        for k in seeds:
            x = Q.summary(arm, k)
            if x and x.get("hard_final") is not None and x.get("hard_base") is not None:
                n_u += 1
                low_u += not x["hard_final"] > x["hard_base"]
    if n and n_u:
        from scipy import stats
        out["pooled_contrast"] = {"uncorrected_W8A8_FLA": [low, n], "unrounded": [low_u, n_u],
                                  "p_one_sided": stats.fisher_exact([[low, n - low], [low_u, n_u - low_u]], alternative="greater")[1]}
    (LB / "results" / "rlul_verdict.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
