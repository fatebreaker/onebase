"""CO-SERVE-MATH-LOAD (docs/REGISTRY.md, registered before any data; descriptive, with one registered bound; written before any data):
CO-SERVE-MATH's co-served 9B math recipe on one Hopper GPU at 0, 1, 2, 4 and 8 tenant requests/s, 20 RL steps each (mlsys
jobs/pod_runs/coserve_load2_all.sh -> $SCRATCH/coserve/run_mathload_<tag>_r<rate>).
Per rate: the RL step time (mean over steps 3-20) and its ratio to rate 0's; the tenants' median TPOT and p99 TTFT alone (A chunks)
and beside RL (the Y chunks inside the RL phase), the share of tenant requests completed; GPU-hours of the co-located run against
two GPUs (RL alone at rate 0, plus one GPU serving the tenants).
Bound (registered): the highest rate at which the tenants beside RL keep CO-SERVE-MATH's bounds (median TPOT <= 100 ms, p99 TTFT
<= 5 s, >= 99.9% completed) and RL's step time stays <= 1.5x rate 0's is reported as the co-location limit on this GPU.
    python3 bench/score_coserve_mathload.py [--tag h200] [--json results/coserve_mathload_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_coserve import phases, requests  # noqa: E402
from score_coserve_math import scored, segments  # noqa: E402

CS = Path("$SCRATCH/coserve")
AD = Path("$SCRATCH/std_bench/adapters")
RATES = (0, 1, 2, 4, 8)


def one(tag: str, r: int) -> dict | None:
    d, ad = CS / f"run_mathload_{tag}_r{r}", AD / f"qwen9_dapo_CSLOAD_{tag}_r{r}"
    if not (d / "phases.jsonl").exists() or not (ad / "log.jsonl").exists():
        return None
    rs = runlog.read(ad / "log.jsonl")
    late = [x for x in rs if x["step"] >= 3]
    out = {"steps": rs[-1]["step"] if rs else 0, "t_step": st.fmean(x["t"] for x in late) if late else None,
           "t_gen": st.fmean(x["t_gen"] for x in late) if late else None}
    ph, segs = phases(d), segments(d)
    if segs:
        out["rl_phase_h"] = sum(hi - lo for lo, hi, _ in segs) / 3600
    if r:
        an = sorted((k for k in ph if k.startswith("A_") and "end" in ph[k]), key=lambda k: int(k[2:]))
        out["A"], out["Y"] = requests(d, an), requests(d, scored(ph, segs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h200")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs = {r: one(a.tag, r) for r in RATES}
    base = runs[0]["t_step"] if runs.get(0) and runs[0].get("t_step") else None
    lim = None
    for r in RATES[1:]:
        x = runs.get(r)
        if not x or x.get("t_step") is None or "Y" not in x:
            continue
        y = x["Y"]
        x["rl_ratio"] = x["t_step"] / base if base else None
        x["within"] = bool(base and y.get("tpot_p50", 1e9) <= 100 and y.get("ttft_p99", 1e9) <= 5000
                           and y["completed"] / max(y["sent"], 1) >= 0.999 and x["rl_ratio"] <= 1.5)
        x["Y_share_completed"] = y["completed"] / max(y["sent"], 1)
        if x["within"]:
            lim = r
    out = {"tag": a.tag, "runs": runs, "co_location_limit_req_s": lim}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
