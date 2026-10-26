"""RL-LONG-PROBE -SHBF (docs/REGISTRY.md, registered before any data of its arms; written before any data): one bf16 copy shared by
the bf16 engine and trainer (SB9L, SB27) and FlashRL's recipe with the trainer's bf16 base offloaded (F9L, F27) at RL-LONG-PROBE's
settings (thinking mode, at most 8,192 new tokens, 12 steps, steps 3-12 scored; bench/score_rlprobe.py's arithmetic), against the
probe's FP8 placements (K: one FP8 copy releasing its KV cache; O: FP8 trainer offloaded) and its bf16 offload arm (B).
  (i) for each model, K's mean step time at most 0.85x SB's, or SB runs out of memory (one bf16 copy does not fit 8,192-token rollouts)
Descriptive: SB's and F's step and generation times against K, O and B; their peak host memory (summed Pss of the run's process group,
logs/rlprobe/<run>_pss.jsonl); their KV cache.
    python3 bench/score_rlprobe_shbf.py [--json results/rlprobe_shbf_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_rlprobe import arm  # noqa: E402

MODELS = {"9B": ("SB9L", "F9L", "K9L", "O9L", "B9L"), "27B": ("SB27", "F27", "K27", "O27", "B27")}


def peak_pss(code: str) -> float | None:
    f = LB / "logs/rlprobe" / f"probe8k_{code}_s0_pss.jsonl"
    v = [json.loads(x)["pss_gib"] for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    return max(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for m, (sb, f, k, o, b) in MODELS.items():
        X = {c: arm(c) for c in (sb, f, k, o, b)}
        for c in (sb, f):
            X[c]["peak_pss_gib"] = peak_pss(c)
        rel = {c: {r: X[c]["t"] / X[r]["t"] for r in (k, o, b) if "t" in X[r]} for c in (sb, f) if "t" in X[c]}
        v: dict = {}
        if "t" in X[sb] and "t" in X[k]:
            ratio = X[k]["t"] / X[sb]["t"]
            v["i"] = {"K_over_SB_step": ratio, "met": ratio <= 0.85}
        elif X[sb]["oom"] and X[sb]["steps"] < 12:
            v["i"] = {"SB_out_of_memory_at_step": X[sb]["steps"] + 1, "met": True}
        res[m] = {"arms": X, "relative": rel, "verdict": v}
        print(m, json.dumps({c: {kk: (round(y, 2) if isinstance(y, float) else y) for kk, y in X[c].items()} for c in (sb, f)}))
        print(m, "relative:", json.dumps({c: {r: round(y, 3) for r, y in d.items()} for c, d in rel.items()}), "verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
