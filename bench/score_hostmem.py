"""RL-HOSTMEM (docs/REGISTRY.md, registered before any data; a measurement): the host memory each placement of the trainer's copy
needs, Qwen3.8-27B's recipe on one H100 (mlsys jobs/pod_runs/hostmem_pod.sh), 4 steps per arm. From results/hostmem/<arm>.jsonl
(the run's process group's Pss every 2 s) and <adapters>/hostmem/hostmem_<arm>_s0/log.jsonl (step times; bench/runlog.py):
  (i) over steps 2-4 (median of the samples between the end of step 1 and the end of step 4), O27's host memory exceeds S27's and
      K27's by at least 20 GiB, and B27's exceeds O27's by at least 15 GiB
Descriptive: each arm's peak (loading included), its median over steps 2-4, its mean step time, and how many of its runs fit a
250 GB pod share and a 1 TB node by the peak.
-9B extension (registered before its data; mlsys jobs/pod_runs/hostmem9_pod.sh, Qwen3.5-9B on one L40S, arms B9 O9 T9 S9 K9):
  (i-9B) O9's host memory over steps 2-4 exceeds S9's and K9's by at least 8 GiB, and B9's exceeds O9's by at least 5 GiB
    python3 bench/score_hostmem.py [--json results/hostmem_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402

AD = Path("$SCRATCH/std_bench/adapters/hostmem")
ARMS = ("B27", "O27", "T27", "S27", "K27", "B9", "O9", "T9", "S9", "K9")
POD_GB, NODE_GB = 250, 1000


def arm(code: str) -> dict:
    f = LB / f"results/hostmem/{code}.jsonl"
    smp = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    rs = runlog.read(AD / f"hostmem_{code}_s0" / "log.jsonl")
    out = {"samples": len(smp), "steps": rs[-1]["step"] if rs else 0}
    if not smp:
        return out
    peak = max(x["pss_gib"] for x in smp)
    out.update(peak_gib=peak, fit_pod=int(POD_GB / (peak * 1.073741824)), fit_node=int(NODE_GB / (peak * 1.073741824)))
    if rs and rs[-1]["step"] >= 4:
        # step k ends at the sum of the first k step times, counted back from the run's last sample (the sampler runs to the end)
        t_end = smp[-1]["t"]
        total = sum(r["t"] for r in rs[:4])
        t1 = t_end - (total - rs[0]["t"])            # end of step 1, approximately (loading and the final save are not timed)
        window = [x["pss_gib"] for x in smp if t1 <= x["t"] <= t_end]
        out.update(steady_gib=st.median(window) if window else None, step_s=st.fmean(r["t"] for r in rs[1:4]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {c: arm(c) for c in ARMS}
    for c, x in res.items():
        print(c, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in x.items()}))
    s = {c: res[c].get("steady_gib") for c in ARMS}
    v = {}
    for tag, sfx, d_os, d_bo in (("i", "27", 20, 15), ("i-9B", "9", 8, 5)):
        O, S, K, B = (s[f"{a}{sfx}"] for a in "OSKB")
        if all(x is not None for x in (O, S, K, B)):
            v[tag] = {"O_minus_S": O - S, "O_minus_K": O - K, "B_minus_O": B - O,
                      "met": O - max(S, K) >= d_os and B - O >= d_bo}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"arms": res, "verdict": v}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
