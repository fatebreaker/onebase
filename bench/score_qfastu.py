"""Score RL-UNCORRECTED-FAST (docs/REGISTRY.md, registered 2026-10-23 06:33 UTC; this scorer written 10-23 before any S8UF data).

Arm S8UF: S8TF's recipe on the H100 (W8A8 engine with the plugin, the weight-matched FP8 trainer with flash-linear-attention)
without TIS, seeds 0-5, each evaluated as S8TF's (its own FP8 engine, greedy; OlympiadBench and Minerva pooled).
  (i)  every S8UF run's mean mismatch over steps 151-200 is at least 2x S8MH's (the matched engine beside the same trainer, no
       correction; S8MH's runs pooled over the same steps).
Descriptive: each run's final accuracy and whether it ends at or below its own pre-RL accuracy; S8UF minus S0TF and minus S8TF
(Welch over runs, and paired by seed); the collapse contrast holding the trainer fixed: S8UF's runs at or below pre-RL accuracy
against the unrounded engines' uncorrected runs beside the same trainer (S8MH, S8ML, S8MS9L, Marlin S8WA), one-sided Fisher.
Stopping: runs trained and evaluated by 2026-10-26 12:00 UTC.
    python3 bench/score_qfastu.py -> results/qfastu_verdict.json"""
from __future__ import annotations

import datetime as dt
import json
import math
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

for _a in ("S8UF", "S8TF", "S8MH", "S8ML", "S8MS9L", "S8WA"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")
STOP = dt.datetime(2026, 10, 26, 12, 0, tzinfo=dt.timezone.utc)
UNROUNDED = {"S8MH": range(6), "S8ML": range(3), "S8MS9L": range(5), "S8WA": range(4)}


def late_mismatch(arm: str, s: int) -> float | None:
    rs = Q.rows(arm, s)
    v = [r["mismatch"] for r in rs if r.get("mismatch") is not None and 151 <= r["step"] <= 200]
    return st.fmean(v) if v and rs[-1]["step"] >= 200 else None


def paired(a: dict, b: dict) -> dict | None:
    ks = sorted(set(a) & set(b))
    if len(ks) < 2:
        return None
    from scipy import stats
    d = [a[k] - b[k] for k in ks]
    m, se = st.fmean(d), st.stdev(d) / math.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    return {"seeds": ks, "diff": m, "ci95": [m - h, m + h]}


def main() -> int:
    runs = {arm: {k: Q.summary(arm, k) for k in range(6)} for arm in ("S8UF", "S8TF", "S0TF")}
    done = {arm: {k: x for k, x in rs.items() if x and x.get("hard_final") is not None} for arm, rs in runs.items()}
    out: dict = {"seeds": {arm: sorted(d) for arm, d in done.items()},
                 "final": len(done["S8UF"]) == 6 or dt.datetime.now(dt.timezone.utc) >= STOP}
    mh = [m for k in range(6) if (m := late_mismatch("S8MH", k)) is not None]
    uf = {k: m for k in range(6) if (m := late_mismatch("S8UF", k)) is not None}
    if mh and uf:
        ref = st.fmean(mh)
        out["i"] = {"S8MH_late": ref, "S8UF_late": uf, "ratios": {k: v / ref for k, v in uf.items()},
                    "verdict": "MET" if all(v >= 2 * ref for v in uf.values()) else "NOT MET"}
    out["runs"] = {k: {"final": x["hard_final"], "base": x["hard_base"], "at_or_below_base": not x["hard_final"] > x["hard_base"],
                       "gpu_hours": round(x["gpu_hours"], 3)} for k, x in done["S8UF"].items()}
    acc = {arm: {k: x["hard_final"] for k, x in d.items()} for arm, d in done.items()}
    for ref in ("S0TF", "S8TF"):
        a, b = list(acc["S8UF"].values()), list(acc[ref].values())
        out[f"S8UF_minus_{ref}"] = {"welch": welch(a, b) if len(a) >= 2 and len(b) >= 2 else None,
                                    "paired": paired(acc["S8UF"], acc[ref])}
    low_uf = sum(1 for x in out["runs"].values() if x["at_or_below_base"])
    low_un, n_un = 0, 0
    for arm, seeds in UNROUNDED.items():
        for k in seeds:
            x = Q.summary(arm, k)
            if x and x.get("hard_final") is not None and x.get("hard_base") is not None:
                n_un += 1
                low_un += not x["hard_final"] > x["hard_base"]
    if out["runs"] and n_un:
        from scipy import stats
        p = stats.fisher_exact([[low_uf, len(out["runs"]) - low_uf], [low_un, n_un - low_un]], alternative="greater")[1]
        out["collapse_contrast"] = {"S8UF": [low_uf, len(out["runs"])], "unrounded_same_trainer": [low_un, n_un], "p_one_sided": p}
    (LB / "results" / "qfastu_verdict.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
