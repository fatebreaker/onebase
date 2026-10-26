"""ICML-E2B (docs/REGISTRY.md, registered before any data): ICML-E2-PILOT's fake-quant engines (S8FQT to nearest, S8FST
stochastically) with truncated importance sampling (C = 2), S8MH's recipe otherwise, seeds 0-2, from the runs' logs
(<adapters>/qwen9_dapo_<arm>_s<k>/log.jsonl, one row per step) and their bf16-engine evaluations (results/std_bench/<run>/
bf16_{base,exact}).
  (i)   every run completes its 200 steps
  (ii)  every run ends above its own pre-RL accuracy (OlympiadBench+Minerva pooled, bf16 engine): no collapse
  (iii) S8FST: mean mismatch over steps 181-200 at most 2x its mean over steps 1-10 (no runaway; the pilot's S8FS: 81-86x)
Descriptive: final accuracy S8FST minus S8FQT (Welch), the TIS weights' mean and clipped share, the top-32 TV/KL diagnostics
early and late, GPU-hours.
    python3 bench/score_e2b.py [--json results/std_bench/e2b_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_e2pilot as E  # noqa: E402
from score_rlmulti import welch  # noqa: E402

ARMS, SEEDS = ("S8FQT", "S8FST"), (0, 1, 2)


def summary(arm: str, k: int) -> dict:
    out = E.summary(arm, k)
    rs = E.rows(out["run"])
    tis = [r.get("tis_mean") for r in rs if r.get("tis_mean") is not None]
    clip = [r.get("tis_clip") for r in rs if r.get("tis_clip") is not None]
    out.update(tis_mean=st.fmean(tis) if tis else None, tis_clip=st.fmean(clip) if clip else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: [summary(arm, k) for k in SEEDS] for arm in ARMS}
    for xs in R.values():
        for x in xs:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v: dict = {}
    complete = [x for xs in R.values() for x in xs if x["done"] and x["steps"] >= 200 and x["final"] is not None]
    if len(complete) == len(ARMS) * len(SEEDS):
        v["i"] = {"met": True}
        v["ii"] = {"final_minus_base": {x["run"]: x["final"] - x["base"] for x in complete},
                   "met": all(x["final"] > x["base"] for x in complete)}
        s = R["S8FST"]
        v["iii"] = {"ratio": [x["mismatch_last20"] / x["mismatch_first10"] for x in s],
                    "met": all(x["mismatch_last20"] <= 2 * x["mismatch_first10"] for x in s)}
        v["descriptive_SR_minus_RTN"] = welch([x["final"] for x in R["S8FST"]], [x["final"] for x in R["S8FQT"]])
    else:
        v["status"] = f"incomplete ({len(complete)} of {len(ARMS) * len(SEEDS)} runs trained and evaluated)"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
