"""E2E-QWEN3-TUNED-H100 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v34 M5): E2E-
QWEN3-TUNED-H200's design and prediction on one H100 (mlsys jobs/pod_runs/qwen3tuned_h100_pod.sh -> results/e2e_qwen3tuned_h100.jsonl),
scored by bench/score_qwen3tuned_h200.py's logic on this file; means of the three measured starts.
  (i) W1(plugin-pdl+r64:n1) < min(W1 of stock-tuned, stock-tuned-torch, stock-dual-tuned at 64:1)
Descriptive: W2 at 64:32 of the plugin against the best of the three.
    python3 bench/score_qwen3tuned_h100.py [--json results/qwen3tuned_h100_verdict.json]"""
from __future__ import annotations

import importlib.util
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_qwen3tuned_h200", LB / "bench/score_qwen3tuned_h200.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
S.F = LB / "results/e2e_qwen3tuned_h100.jsonl"

if __name__ == "__main__":
    raise SystemExit(S.main())
