"""RL-72B-ONE-GPU (docs/REGISTRY.md, registered before any data; this scorer written before any data): LoRA RL of Qwen2.5-72B-Instruct
on one H200 by placement of the trainer's base (mlsys jobs/pod_runs/rl72_pod.sh -> <adapters>/rl72/qwen72_dapo_<ARM>_s0, logs and Pss
traces lowbit logs/rl72/). The 27B's 2,048-token recipe. Per arm: whether its smoke and run completed, out-of-memory or failed engine
starts, mean seconds per step over steps 3-12 (and over the run), peak host memory, the engine's KV cache.
  (i)   K72 (one FP8 copy releasing its KV cache) completes its 100 steps on one H200
  (ii)  each placement holding a bf16 copy or two copies (T72, F72, B72, SB72) fails on one H200 (out of GPU memory at load, at the
        engine's start or in a step) before completing its 12 steps
  (iii) K72's peak host memory at most 0.25x O72's (the FP8 trainer offloaded)
Descriptive: K72's step time against O72's; K72's final accuracy (MATH-500, OlympiadBench, Minerva, its own FP8 engine) against the
base model's on the same engine.
    python3 bench/score_rl72.py [--json results/std_bench/rl72_verdict.json]"""
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

AD = Path("$SCRATCH/std_bench/adapters/rl72")
LOGD = LB / "logs/rl72"
RES = LB / "results/std_bench"
FAIL = re.compile(r"out of memory|OutOfMemory|No available memory for the cache blocks|insufficient memory|CUDA error: out of memory")


def arm(code: str) -> dict:
    name = f"qwen72_dapo_{code}_s0"
    f = AD / name / "log.jsonl"
    rs = runlog.read(f) if f.exists() else []
    out: dict = {"steps": rs[-1]["step"] if rs else 0}
    texts = [p.read_text(errors="replace") for p in (LOGD / f"{name}_smoke.out", LOGD / f"{name}.out") if p.exists()]
    out["ran"] = bool(texts)
    out["oom"] = any(FAIL.search(t) for t in texts)
    kv = [m for t in texts for m in re.findall(r"Available KV cache memory: ([0-9.]+) GiB", t)]
    out["kv_gib"] = float(kv[-1]) if kv else None
    if rs:
        late = [r for r in rs if 3 <= r["step"] <= 12]
        if late:
            out["t_3_12"] = st.fmean(r["t"] for r in late)
            out["t_gen_3_12"] = st.fmean(r["t_gen"] for r in late)
        out["t_all"] = st.fmean(r["t"] for r in rs)
        out["gpu_hours"] = sum(r["t"] for r in rs) / 3600
    peaks = []
    for p in (LOGD / f"{name}_smoke_pss.jsonl", LOGD / f"{name}_pss.jsonl"):
        if p.exists():
            peaks += [json.loads(x)["pss_gib"] for x in p.read_text().splitlines() if x.strip()]
    out["peak_pss_gib"] = max(peaks) if peaks else None
    return out


def hard(name: str, variant: str) -> float | None:
    vals = []
    for b in ("olympiad", "minerva"):
        f = RES / name / variant / f"{b}.jsonl"
        if not f.exists():
            return None
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines() if x.strip()]
    return 100 * sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    X = {c: arm(c) for c in ("K72", "O72", "T72", "F72", "B72", "SB72")}
    v: dict = {}
    if X["K72"]["ran"]:
        v["i"] = {"steps": X["K72"]["steps"], "met": X["K72"]["steps"] >= 100}
    fails = ("T72", "F72", "B72", "SB72")
    if all(X[c]["ran"] for c in fails):
        each = {c: {"steps": X[c]["steps"], "failed": X[c]["steps"] < 12, "oom": X[c]["oom"]} for c in fails}
        v["ii"] = {"arms": each, "met": all(e["failed"] for e in each.values())}
    k, o = X["K72"]["peak_pss_gib"], X["O72"]["peak_pss_gib"]
    if k and o and X["O72"]["steps"] >= 12:
        v["iii"] = {"K_gib": k, "O_gib": o, "ratio": k / o, "met": k <= 0.25 * o}
    desc = {}
    if "t_3_12" in X["K72"] and "t_3_12" in X["O72"]:
        desc["K_over_O_step_3_12"] = X["K72"]["t_3_12"] / X["O72"]["t_3_12"]
    nm = "qwen72_dapo_K72_s0"
    desc["hard_final"] = hard(nm, "fp8eng_exact")
    desc["hard_base"] = hard(nm, "fp8eng_base")
    out = {"arms": X, "verdict": v, "descriptive": desc}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
