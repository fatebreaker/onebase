"""RL-LONG-16K (docs/REGISTRY.md, registered before any data; this scorer written before any data): the 27B's placements at
16,384-token rollouts on one H100 (thinking mode, Q27's recipe, 12 steps, steps 3-12 scored, seed 0; mlsys
jobs/pod_runs/rlprobe16_pod.sh). Per arm, from its log (<adapters>/probe16k/probe16k_<ARM>_s0/log.jsonl, bench/runlog.py), its run
log and its Pss trace (lowbit logs/rlprobe16/): steps completed, whether it ran out of memory, mean seconds per step and per step
generating over steps 3-12, response length, the engine's KV cache, the trainer's peak, the peak host memory.
  (i)   K27X (one FP8 copy releasing its KV cache) completes its 12 steps
  (ii)  each placement with a bf16 trainer (F27X, B27X, SB27X) runs out of GPU memory before completing its 12 steps
  (iii) K27X's mean step time at most 1.00x O27X's (the FP8 trainer offloaded)
  (iv)  K27X's peak host memory at most 0.25x O27X's
A prediction whose arms did not run is not scored.
    python3 bench/score_rlprobe16.py [--json results/rlprobe16_verdict.json]"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402

AD = Path("$SCRATCH/std_bench/adapters/probe16k")
LOGD = LB / "logs/rlprobe16"
STEPS, FIRST = 12, 3
ARMS = ("K27X", "F27X", "B27X", "SB27X", "O27X")
BF16_TRAINER = ("F27X", "B27X", "SB27X")


def arm(code: str) -> dict:
    name = f"probe16k_{code}_s0"
    f = AD / name / "log.jsonl"
    rs = runlog.read(f) if f.exists() else []
    log = LOGD / f"{name}.out"
    t = log.read_text(errors="replace") if log.exists() else ""
    kv = re.search(r"Available KV cache memory: ([0-9.]+) GiB", t)
    out = {"ran": bool(t), "steps": rs[-1]["step"] if rs else 0, "kv_gib": float(kv.group(1)) if kv else None,
           "oom": bool(re.search(r"out of memory|OutOfMemory", t))}
    peaks = [r["trainer_peak_gib"] for r in rs if r.get("trainer_peak_gib") is not None]
    out["trainer_peak_gib"] = max(peaks) if peaks else None
    use = [r for r in rs if r["step"] >= FIRST]
    if rs and rs[-1]["step"] >= STEPS and use:
        out.update(t=st.fmean(r["t"] for r in use), t_gen=st.fmean(r["t_gen"] for r in use),
                   tokens=st.fmean(r.get("tokens", 0) for r in use), len=st.fmean(r.get("len", 0) for r in use),
                   gen_tok_s=sum(r.get("tokens", 0) for r in use) / sum(r["t_gen"] for r in use))
    p = LOGD / f"{name}_pss.jsonl"
    v = [json.loads(x)["pss_gib"] for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
    out["peak_pss_gib"] = max(v) if v else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    X = {c: arm(c) for c in ARMS}
    K, O = X["K27X"], X["O27X"]
    v: dict = {}
    if K["ran"]:
        v["i"] = {"steps": K["steps"], "met": K["steps"] >= STEPS}
    ran = [c for c in BF16_TRAINER if X[c]["ran"]]
    if len(ran) == len(BF16_TRAINER):
        each = {c: {"steps": X[c]["steps"], "oom": X[c]["oom"], "met": X[c]["oom"] and X[c]["steps"] < STEPS} for c in ran}
        v["ii"] = {"arms": each, "met": all(e["met"] for e in each.values())}
    if "t" in K and "t" in O:
        v["iii"] = {"K_over_O_step": K["t"] / O["t"], "met": K["t"] <= O["t"]}
    if K["peak_pss_gib"] and O["peak_pss_gib"] and K["steps"] >= STEPS and O["steps"] >= STEPS:
        r = K["peak_pss_gib"] / O["peak_pss_gib"]
        v["iv"] = {"K_gib": K["peak_pss_gib"], "O_gib": O["peak_pss_gib"], "ratio": r, "met": r <= 0.25}
    for c, x in X.items():
        print(c, json.dumps({k: (round(y, 2) if isinstance(y, float) else y) for k, y in x.items()}))
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"arms": X, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
