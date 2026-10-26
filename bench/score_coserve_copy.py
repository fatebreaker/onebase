"""CO-SERVE-COPY (docs/REGISTRY.md; this scorer written before any data): CO-SERVE's FP8 design with the trainer holding its own
FP8 copy (mode S8) instead of mapping the engine's (S9). Applies the registered predictions to bench/score_coserve.py's output for
this run (unchanged) against CO-SERVE's (results/coserve_verdict.json, run1):
  (i)   both trainer runs complete their 12 steps with no step skipped for memory, and every scored tenant request completes
        (score_coserve's (i))
  (ii)  peak GPU memory during RL alone (X) at least 8 GiB above CO-SERVE's (what one shared copy saves)
  (iii) RL step time with tenants (Y, steps 3-12) between 0.9x and 1.1x CO-SERVE's
  (iv)  the tenants' median TPOT beside RL between 0.9x and 1.1x CO-SERVE's
Descriptive: every score_coserve quantity side by side, the RL slowdown (Y over X), GPU-hours colocated and separate.
    python3 bench/score_coserve.py --dir <OUTD> --json results/coserve_copy_run.json
    python3 bench/score_coserve_copy.py [--run results/coserve_copy_run.json] [--json results/coserve_copy_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def peak(mem: dict, phase: str) -> float | None:
    v = mem.get(phase)
    if isinstance(v, dict):
        v = v.get("mem_peak_mib", v.get("peak_mib"))
    return float(v) if isinstance(v, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/coserve_copy_run.json")
    ap.add_argument("--ref", default="results/coserve_verdict.json")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rp, fp = LB / a.run, LB / a.ref
    if not rp.exists():
        print(json.dumps({"status": "incomplete", "missing": a.run}))
        return 0
    run, ref = json.loads(rp.read_text()), json.loads(fp.read_text())
    out: dict = {"run": a.run, "ref": a.ref, "side_by_side": {}}
    for k in ("A", "Y", "X_rl", "Y_rl", "gpu_hours", "memory", "wall_s"):
        out["side_by_side"][k] = {"copy": run.get(k), "shared": ref.get(k)}
    v: dict = {"i": run.get("verdict", {}).get("i", "incomplete")}
    px, rx = peak(run.get("memory", {}), "rl_x"), peak(ref.get("memory", {}), "rl_x")
    if px is None or rx is None:
        v["ii"] = "incomplete"
    else:
        out["peak_x_gib"] = {"copy": px / 1024, "shared": rx / 1024, "diff": (px - rx) / 1024}
        v["ii"] = "met" if px - rx >= 8 * 1024 else "not met"
    ty, ry = (run.get("Y_rl") or {}).get("t_step"), (ref.get("Y_rl") or {}).get("t_step")
    if ty is None or ry is None:
        v["iii"] = "incomplete"
    else:
        out["rl_y_step_ratio"] = ty / ry
        v["iii"] = "met" if 0.9 <= ty / ry <= 1.1 else "not met"
    tt, rt = (run.get("Y") or {}).get("tpot_p50"), (ref.get("Y") or {}).get("tpot_p50")
    if tt is None or rt is None:
        v["iv"] = "incomplete"
    else:
        out["tpot_y_ratio"] = tt / rt
        v["iv"] = "met" if 0.9 <= tt / rt <= 1.1 else "not met"
    xr, yr = run.get("X_rl", {}).get("t_step"), run.get("Y_rl", {}).get("t_step")
    if xr and yr:
        out["rl_slowdown_copy"] = yr / xr
    out["verdict"] = v
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
