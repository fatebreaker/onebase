"""RL-INTEGRATED-KV (docs/REGISTRY.md, registered before its data; this scorer written before any data; mock review R2 v46, M3): a
30-step systems probe of the matched engine with one shared FP8 copy releasing its KV cache while the trainer runs (arm S8MK9L, mlsys
jobs/pod_runs/qrlintkv_pod.sh, one L40S, seed 0), against the first 30 steps of RL-INTEGRATED's S8MS9L (the same recipe, KV kept, share
0.55) and RL-KVSLEEP's S8K9L (W8A8 with TIS, KV released, share 0.80), each over its seeds 0-2.
  (i)  S8MK9L completes its 30 steps
Descriptive: seconds per step (total and generating) over steps 3-30, the per-token mismatch over steps 1-30, the engine's KV cache.
    python3 bench/score_rlintkv.py [--json results/std_bench/rlintkv_verdict.json]"""
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

A = Path("$SCRATCH/std_bench/adapters")
STEPS, FIRST = 30, 3


def run(name: str, log: Path | None = None) -> dict:
    rs = [r for r in runlog.read(A / name / "log.jsonl") if r["step"] <= STEPS]
    out: dict = {"steps": rs[-1]["step"] if rs else 0}
    use = [r for r in rs if r["step"] >= FIRST]
    if use:
        out.update(t=st.fmean(r["t"] for r in use), t_gen=st.fmean(r["t_gen"] for r in use))
    mm = [r["mismatch"] for r in rs if r.get("mismatch") is not None]
    out["mismatch"] = st.fmean(mm) if mm else None
    if log is not None and log.exists():
        kv = re.findall(r"Available KV cache memory: ([0-9.]+) GiB", log.read_text(errors="replace"))
        out["kv_gib"] = float(kv[-1]) if kv else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    X = {"S8MK9L": run("qwen9_dapo_S8MK9L_probe_s0", LB / "logs/lb_qrlintkv_qwen9_dapo_S8MK9L_probe_s0.out")}
    for arm in ("S8MS9L", "S8K9L"):
        rs = [run(f"qwen9_dapo_{arm}_s{k}") for k in (0, 1, 2)]
        rs = [r for r in rs if "t" in r]
        X[arm] = {k: st.fmean(r[k] for r in rs) for k in ("t", "t_gen", "mismatch") if rs and all(r.get(k) is not None for r in rs)}
        X[arm]["n"] = len(rs)
    v = {"i": {"steps": X["S8MK9L"]["steps"], "met": X["S8MK9L"]["steps"] >= STEPS}} if X["S8MK9L"]["steps"] else {}
    for c, x in X.items():
        print(c, json.dumps({k: (round(y, 4) if isinstance(y, float) else y) for k, y in x.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"arms": X, "verdict": v}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
