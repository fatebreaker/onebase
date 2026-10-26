"""CO-SERVE-LOAD (docs/REGISTRY.md, registered 2026-10-20 21:00 UTC, before any data): CO-SERVE's design (one H100, engine share
0.50, 16 random rank-16 tenants, phases A, X, Y with the G3 trainer, 12 steps) at 4 and 8 tenant requests/s with the FP8 base, then
the bf16 control (CO-SERVE-BF16's configuration) at 4 requests/s; each run scored by bench/score_coserve.py unchanged
(<CS>/run_load_{fp8_r4,fp8_r8,bf16_r4}).
  (i)   at each rate and base, both trainer runs complete their 12 steps with no step skipped for memory and every scored tenant
        request completes (score_coserve's (i), per run)
  (ii)  FP8 at 4 requests/s, beside RL: the tenants' median TPOT at most 50 ms and p99 TTFT at most 5 s (CO-SERVE's (iii) bounds)
  (iii) bf16 at 4 against FP8 at 4: RL step time with tenants (steps 3-12) at least 1.2x, and the tenants' median TPOT beside RL
        at least 1.2x (CO-SERVE-BF16's (ii)-(iii))
Descriptive: FP8 at 8 requests/s against (ii)'s bounds; the load curve over 2 (CO-SERVE's run1, CO-SERVE-BF16's run_bf16), 4 and
8 requests/s: the tenants' TPOT and TTFT alone and beside RL, the RL slowdown, GPU-hours colocated against a GPU each; peak memory.
    python3 bench/score_coserve_load.py [--json results/coserve_load_verdict.json]"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
CS = Path("$SCRATCH/coserve")
RUNS = {("fp8", 2): "run1", ("bf16", 2): "run_bf16", ("fp8", 4): "run_load_fp8_r4", ("fp8", 8): "run_load_fp8_r8",
        ("bf16", 4): "run_load_bf16_r4"}


def score(run: str) -> dict | None:
    """bench/score_coserve.py's output for one run directory, or None if the run has not ended."""
    d = CS / run
    if not (d / "server_final.log").exists():
        return None
    with tempfile.NamedTemporaryFile(suffix=".json") as t:
        subprocess.run([sys.executable, str(LB / "bench/score_coserve.py"), "--dir", str(d), "--json", t.name], check=True,
                       capture_output=True, timeout=600)
        return json.loads(Path(t.name).read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    s = {k: score(r) for k, r in RUNS.items()}
    need = [("fp8", 4), ("fp8", 8), ("bf16", 4)]
    if any(s[k] is None for k in need):
        print("incomplete:", [RUNS[k] for k in need if s[k] is None])
        return 1
    f4, f8, b4 = (s[k] for k in need)
    t_ratio = b4["Y_rl"]["t_step"] / f4["Y_rl"]["t_step"]
    tpot_ratio = b4["Y"]["tpot_p50"] / f4["Y"]["tpot_p50"]
    v = {"i": {"per_run": {f"{p}_r{r}": s[(p, r)]["verdict"]["i"]["met"] for p, r in need},
               "dropped": {f"{p}_r{r}": s[(p, r)]["Y"]["sent"] - s[(p, r)]["Y"]["completed"]
                           + s[(p, r)]["A"]["sent"] - s[(p, r)]["A"]["completed"] for p, r in need}},
         "ii": {"tpot_p50": f4["Y"]["tpot_p50"], "ttft_p99": f4["Y"]["ttft_p99"], "met": f4["verdict"]["iii"]["met"]},
         "iii": {"step_ratio": t_ratio, "tpot_ratio": tpot_ratio, "met": t_ratio >= 1.2 and tpot_ratio >= 1.2}}
    v["i"]["met"] = all(v["i"]["per_run"].values())
    curve = {}
    for (p, r), x in s.items():
        if x is None:
            continue
        curve[f"{p}_r{r}"] = {"A_tpot_p50": x["A"]["tpot_p50"], "A_ttft_p99": x["A"]["ttft_p99"],
                              "Y_tpot_p50": x["Y"]["tpot_p50"], "Y_ttft_p50": x["Y"]["ttft_p50"], "Y_ttft_p99": x["Y"]["ttft_p99"],
                              "A_tok_s": x["A"]["tok_s"], "Y_tok_s": x["Y"]["tok_s"],
                              "rl_slowdown": x["verdict"]["ii"]["ratio"], "t_step_x": x["X_rl"]["t_step"],
                              "t_step_y": x["Y_rl"]["t_step"], "gpu_h_colocated": x["gpu_hours"]["colocated"],
                              "gpu_h_separate": x["gpu_hours"]["separate"],
                              "mem_peak_mib": {k: m.get("mem_peak_mib") for k, m in x["memory"].items()}}
    for k, c in curve.items():
        print(k, json.dumps({q: (round(y, 3) if isinstance(y, float) else y) for q, y in c.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"curve": curve, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
