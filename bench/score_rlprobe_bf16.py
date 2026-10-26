"""RL-LONG-PROBE -BF16 (docs/REGISTRY.md, registered before any data of its arms): the bf16 offload recipes at long rollouts, B9L
(S0OL's, one L40S) and B27 (S0T27HO's, one H100), against RL-LONG-PROBE's FP8 placements, with bench/score_rlprobe.py's arithmetic
(steps 3-12 of 12, thinking mode, at most 8,192 new tokens).
  (i) K9L's mean step time at most 0.75x B9L's, and K27's at most 0.75x B27's
Descriptive: every FP8 placement's step and generation time against its bf16 arm's.
    python3 bench/score_rlprobe_bf16.py [--json results/rlprobe_bf16_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_rlprobe import arm  # noqa: E402

PAIRS = {"9B": ("B9L", ("O9L", "T9L", "S9L", "K9L")), "27B": ("B27", ("O27", "T27", "S27", "K27"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for m, (b, fp8) in PAIRS.items():
        B = arm(b)
        arms = {c: arm(c) for c in fp8}
        rel = {c: {"step_over_bf16": x["t"] / B["t"], "gen_over_bf16": x["t_gen"] / B["t_gen"]}
               for c, x in arms.items() if "t" in x and "t" in B}
        K = fp8[-1]
        v = {"i": {"ratio": rel[K]["step_over_bf16"], "met": rel[K]["step_over_bf16"] <= 0.75}} if K in rel else {}
        res[m] = {"bf16": B, "relative": rel, "verdict": v}
        print(m, b, json.dumps({k: (round(y, 2) if isinstance(y, float) else y) for k, y in B.items()}))
        print(m, "relative:", json.dumps({c: {k: round(y, 3) for k, y in r.items()} for c, r in rel.items()}), "verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
