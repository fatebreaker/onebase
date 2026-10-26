"""RL-LONG-16K -CHUNK (docs/REGISTRY.md, registered before its data; this scorer written before any data; mock review R2 v46, M4):
RL-LONG-16K's two bf16-trainer offloading arms again with chunked log-probs (train/chunked_head.py: the LM head in recomputed chunks,
no full [length, vocabulary] logits): F27C = F27X (FlashRL's recipe, grpo_q19.py --chunked-head), B27C = B27X (bf16 offloading,
grpo_q20.py --chunked-head), the protocol otherwise unchanged (mlsys jobs/pod_runs/rlprobe16c_pod.sh; 12 steps, 3-12 scored).
  (i)  each of F27C and B27C runs out of GPU memory before completing its 12 steps (the paper's 16K claim, that only an FP8 trainer
       fits, survives chunking)
Descriptive: steps, step and generation times against RL-LONG-16K's K27X and O27X, trainer peaks, peak host memory.
    python3 bench/score_rlprobe16c.py [--json results/rlprobe16c_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_rlprobe16 import STEPS, arm  # noqa: E402

ARMS = ("F27C", "B27C")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    X = {c: arm(c) for c in ARMS + ("K27X", "O27X")}
    v: dict = {}
    if all(X[c]["ran"] for c in ARMS):
        each = {c: {"steps": X[c]["steps"], "oom": X[c]["oom"], "met": X[c]["oom"] and X[c]["steps"] < STEPS} for c in ARMS}
        v["i"] = {"arms": each, "met": all(e["met"] for e in each.values())}
    for c in ARMS:
        if "t" in X[c]:
            v[f"{c}_over_K27X_step"] = X[c]["t"] / X["K27X"]["t"] if "t" in X["K27X"] else None
            v[f"{c}_over_O27X_step"] = X[c]["t"] / X["O27X"]["t"] if "t" in X["O27X"] else None
    for c, x in X.items():
        print(c, json.dumps({k: (round(y, 2) if isinstance(y, float) else y) for k, y in x.items()}))
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"arms": X, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
