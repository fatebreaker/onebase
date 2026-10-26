"""RL-LONG-27B (docs/REGISTRY.md, registered before any data; this scorer written before any data): the 27B's math RL at 8,192-token
rollouts over 100 steps on one H100, seed 0 (mlsys jobs/pod_runs/rlong27_pod.sh): K27L (one FP8 copy releasing its KV cache while the
trainer runs) against B27L (bf16 engine and trainer, base offloaded). Two FP8 copies of the 27B do not fit at these rollouts
(RL-LONG-PROBE's T27 ran out of memory in its first step).
  (i)   K27L completes its 100 steps
  (ii)  K27L's GPU-hours (the sum of its logged step times) at most 0.75x B27L's
  (iii) K27L's peak host memory (summed Pss of the run's process group, logs/rlong27/<run>_pss.jsonl) at most 0.25x B27L's
Descriptive: step and generation times, response lengths, the trainer-engine mismatch, final accuracy (OlympiadBench + Minerva pooled,
each policy on its own engine, non-thinking at 2,048 tokens; one seed each). Stopping: trained by 2026-10-27 12:00 UTC.
    python3 bench/score_rlong27.py [--json results/std_bench/rlong27_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402

AD = Path("$SCRATCH/std_bench/adapters")
RES = LB / "results/std_bench"
EXACT = {"K27L": "fp8eng_exact", "B27L": "bf16_exact"}    # each policy on its own engine


def hard(name: str, variant: str) -> float | None:
    vals = []
    for b in ("olympiad", "minerva"):
        f = RES / name / variant / f"{b}.jsonl"
        if not f.exists():
            return None
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines() if x.strip()]
    return 100 * sum(vals) / len(vals) if vals else None


def arm(code: str) -> dict:
    name = f"qwen27_dapo_{code}_s0"
    rs = runlog.read(AD / name / "log.jsonl")
    out: dict = {"steps": rs[-1]["step"] if rs else 0}
    if rs:
        out["gpu_hours"] = sum(r["t"] for r in rs) / 3600
        late = [r for r in rs if r["step"] >= 3]
        if late:
            out.update(t=st.fmean(r["t"] for r in late), t_gen=st.fmean(r["t_gen"] for r in late),
                       len=st.fmean(r.get("len", 0) for r in late),
                       mismatch=st.fmean(r["mismatch"] for r in late if r.get("mismatch") is not None) if any(r.get("mismatch") is not None for r in late) else None)
    f = LB / "logs/rlong27" / f"{name}_pss.jsonl"
    v = [json.loads(x)["pss_gib"] for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    out["peak_pss_gib"] = max(v) if v else None
    out["hard_final"] = hard(name, EXACT[code])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    K, B = arm("K27L"), arm("B27L")
    v: dict = {"i": {"steps": K["steps"], "met": K["steps"] >= 100}}
    if K["steps"] >= 100 and B["steps"] >= 100:
        r = K["gpu_hours"] / B["gpu_hours"]
        v["ii"] = {"K_h": K["gpu_hours"], "B_h": B["gpu_hours"], "ratio": r, "met": r <= 0.75}
        if K["peak_pss_gib"] and B["peak_pss_gib"]:
            m = K["peak_pss_gib"] / B["peak_pss_gib"]
            v["iii"] = {"K_gib": K["peak_pss_gib"], "B_gib": B["peak_pss_gib"], "ratio": m, "met": m <= 0.25}
    else:
        v["status"] = "incomplete"
    out = {"K27L": K, "B27L": B, "verdict": v}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
