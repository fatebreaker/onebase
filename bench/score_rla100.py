"""RL-A100-BF16 (docs/REGISTRY.md, registered 2026-10-19 04:06 UTC, before any data; this scorer written at its first judgment,
after S0G seeds 5-6's data, implementing the registered rule mechanically): exact bf16 LoRA RL on the A100 as the same-GPU
comparator of RL-WONLY-A100's weight-only runs.
  (i)   S0G seeds 5 and 6 complete their 200 steps
  (ii)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy (each on its own engine), S8WA (seeds 0-3) minus S0G's
        A100 seeds (3-6), within +-5 points; else inconclusive
Descriptive: the A100 bf16 runs' mismatch over training against S8WA's; GPU-hours.
    python3 bench/score_rla100.py [--json results/std_bench/rla100_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

Q.EV["S8WA"] = ("fp8eng_base", "fp8eng_exact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    s0 = {k: Q.summary("S0G", k) for k in (3, 4, 5, 6)}
    wa = {k: Q.summary("S8WA", k) for k in (0, 1, 2, 3)}
    for name, d in (("S0G", s0), ("S8WA", wa)):
        for k, x in d.items():
            print(name, k, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in (x or {}).items()
                                       if kk in ("hard_base", "hard_final", "gpu_hours", "mismatch_first10", "t_gen_mean")}))
    v = {"i": {"steps": {k: (Q.rows("S0G", k)[-1]["step"] if Q.rows("S0G", k) else 0) for k in (5, 6)}}}
    v["i"]["met"] = all(n >= 200 for n in v["i"]["steps"].values())
    f8 = [x["hard_final"] for x in wa.values() if x and x.get("hard_final") is not None]
    b0 = [x["hard_final"] for x in s0.values() if x and x.get("hard_final") is not None]
    if len(f8) == 4 and len(b0) == 4:
        v["ii"] = welch(f8, b0)
    else:
        v["ii"] = {"status": f"incomplete: S8WA {len(f8)} of 4, S0G A100 {len(b0)} of 4"}
    mm = lambda d: [st.fmean(r["mismatch"] for r in Q.rows(n, k) if r["step"] > 180 and r.get("mismatch") is not None)  # noqa: E731
                    for n, k in d]
    v["descriptive"] = {"late_mismatch_S0G_A100": mm([("S0G", k) for k in (3, 4, 5, 6)]),
                        "late_mismatch_S8WA": mm([("S8WA", k) for k in (0, 1, 2, 3)]),
                        "gpu_hours_S0G_A100": [x["gpu_hours"] for x in s0.values() if x],
                        "gpu_hours_S8WA": [x["gpu_hours"] for x in wa.values() if x]}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S0G_A100": s0, "S8WA": wa, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
