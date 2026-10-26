"""SERVE-LEAN-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): SERVE-LEAN-H100's predictions on
one H200 (mlsys jobs/pod_runs/serve_lean_reps_h200_pod.sh via serve_lean_h200_g1.sh -> results/serve_leanreps_h200_<variant>.jsonl),
scored by bench/score_serve_lean.py's logic with an h200 entry: plugin-pdl-lean against stock-tuned at vLLM's default capture sizes,
32 rank-64 tenants, rates 2, 4, 8, 16, means over the three starts:
  (i)  plugin-pdl-lean's median time per output token below stock-tuned's at every rate
  (ii) its p99 time to first token at most 1.1x stock-tuned's at every rate
    python3 bench/score_serve_lean_h200.py [--json results/serve_lean_h200_verdict.json]"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_serve_lean", LB / "bench/score_serve_lean.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
S.GPUS["h200"] = ("plugin-pdl-lean", ("2", "4", "8", "16"), "serve_reps", "plugin-pdl")   # no SERVE-REPS rows exist for the H200

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--gpu", "h200", *sys.argv[1:]]
    raise SystemExit(S.main())
