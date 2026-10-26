"""RL-LONG-27B extension -FO (docs/REGISTRY.md, registered before any data of its runs; this scorer written before any data): the 27B's
math RL at 8,192-token rollouts over 100 steps on one H100: the KV-releasing FP8 copy (K27L, seeds 0 and 1) against FlashRL's recipe
with the trainer's bf16 base offloaded (F27L) and the FP8 trainer offloaded (O27L), seed 0 each (mlsys jobs/pod_runs/rlong27_pod.sh,
rlong27b_pod.sh).
  (iv)  K27L's GPU-hours (mean over its completed seeds) at most 1.05x F27L's
  (v)   K27L's peak host memory (summed Pss of the run's process group, logs/rlong27/<run>_pss.jsonl) at most 0.15x F27L's
  (vi)  K27L's GPU-hours at most 1.05x O27L's
Descriptive: step and generation times, mismatch, final accuracy (OlympiadBench + Minerva pooled, each policy on its own engine) against
B27L. A run counts once it has completed its 100 steps.
    python3 bench/score_rlong27fo.py [--json results/std_bench/rlong27fo_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_rlong27 import hard  # noqa: E402

AD = Path("$SCRATCH/std_bench/adapters")
EXACT = {"K27L": "fp8eng_exact", "F27L": "fp8eng_exact", "O27L": "fp8eng_exact", "B27L": "bf16_exact"}


def run(code: str, seed: int) -> dict:
    name = f"qwen27_dapo_{code}_s{seed}"
    f = AD / name / "log.jsonl"
    rs = runlog.read(f) if f.exists() else []
    out: dict = {"steps": rs[-1]["step"] if rs else 0}
    if rs:
        out["gpu_hours"] = sum(r["t"] for r in rs) / 3600
        late = [r for r in rs if r["step"] >= 3]
        if late:
            mm = [r["mismatch"] for r in late if r.get("mismatch") is not None]
            out.update(t=st.fmean(r["t"] for r in late), t_gen=st.fmean(r["t_gen"] for r in late),
                       mismatch=st.fmean(mm) if mm else None)
    p = LB / "logs/rlong27" / f"{name}_pss.jsonl"
    v = [json.loads(x)["pss_gib"] for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    out["peak_pss_gib"] = max(v) if v else None
    out["hard_final"] = hard(name, EXACT[code])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {"K27L": {s: run("K27L", s) for s in (0, 1)}, "F27L": {0: run("F27L", 0)}, "O27L": {0: run("O27L", 0)},
         "B27L": {0: run("B27L", 0)}}
    done = {c: [x for x in d.values() if x["steps"] >= 100] for c, d in R.items()}
    v: dict = {}
    if done["K27L"] and done["F27L"]:
        kh = st.fmean(x["gpu_hours"] for x in done["K27L"])
        fh = done["F27L"][0]["gpu_hours"]
        v["iv"] = {"K_h": kh, "F_h": fh, "ratio": kh / fh, "met": kh <= 1.05 * fh, "K_seeds": len(done["K27L"])}
        kp = [x["peak_pss_gib"] for x in done["K27L"] if x["peak_pss_gib"]]
        fp = done["F27L"][0]["peak_pss_gib"]
        if kp and fp:
            v["v"] = {"K_gib": max(kp), "F_gib": fp, "ratio": max(kp) / fp, "met": max(kp) <= 0.15 * fp}
    if done["K27L"] and done["O27L"]:
        kh = st.fmean(x["gpu_hours"] for x in done["K27L"])
        oh = done["O27L"][0]["gpu_hours"]
        v["vi"] = {"K_h": kh, "O_h": oh, "ratio": kh / oh, "met": kh <= 1.05 * oh}
    out = {"runs": R, "verdict": v}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
