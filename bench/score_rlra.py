"""Score RL-ROUNDED-ALIKE (docs/REGISTRY.md; written before any data): arm S8RA, the W8A8 engine with the plugin and the trainer's
W8A8 arithmetic (--fp8-train fused) on one L40S, two copies, no correction, seeds 0-2.
  (i)  every S8RA run's mean trainer-engine mismatch over steps 151-200 is at least 2x the matched engine's (S8ML: W8A16 engine and
       trainer, no correction, the same GPU and recipe; S8ML's runs pooled over the same steps)
Descriptive: each run's late mismatch against S8TLF's (W8A8 engine, W8A16 trainer, TIS) and S8ML's; final accuracy (OlympiadBench +
Minerva pooled, own FP8 engine) minus S0OL's (Welch) and whether any run ends at or below its pre-RL accuracy. Stopping: runs trained
and evaluated by 2026-10-26 12:00 UTC.
    python3 bench/score_rlra.py -> results/rlra_verdict.json"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

for _a in ("S8RA", "S8TLF", "S8ML"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))
Q.EV.setdefault("S0OL", ("bf16_base", "bf16_exact"))


def late(arm: str, s: int) -> float | None:
    rs = Q.rows(arm, s)
    v = [r["mismatch"] for r in rs if r.get("mismatch") is not None and 151 <= r["step"] <= 200]
    return st.fmean(v) if v and rs and rs[-1]["step"] >= 200 else None


def main() -> int:
    out: dict = {}
    lm = {arm: {k: m for k in range(7) if (m := late(arm, k)) is not None} for arm in ("S8RA", "S8ML", "S8TLF")}
    out["late_mismatch"] = lm
    if lm["S8RA"] and lm["S8ML"]:
        ref = st.fmean(lm["S8ML"].values())
        out["i"] = {"S8ML_late": ref, "ratios": {k: v / ref for k, v in lm["S8RA"].items()}, "n": len(lm["S8RA"]),
                    "verdict": "MET" if all(v >= 2 * ref for v in lm["S8RA"].values()) else "NOT MET"}
    acc = {}
    for arm in ("S8RA", "S0OL"):
        acc[arm] = {}
        for k in range(7):
            x = Q.summary(arm, k)
            if x and x.get("hard_final") is not None:
                acc[arm][k] = (x["hard_final"], x.get("hard_base"))
    a = [v[0] for v in acc["S8RA"].values()]
    b = [v[0] for v in acc["S0OL"].values()]
    out["accuracy"] = {"S8RA": acc["S8RA"], "minus_S0OL": welch(a, b) if len(a) >= 2 and len(b) >= 2 else None,
                       "at_or_below_base": [k for k, (f, base) in acc["S8RA"].items() if base is not None and not f > base]}
    (LB / "results" / "rlra_verdict.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
