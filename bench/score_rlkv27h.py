"""RL-KVSLEEP-H100 (docs/REGISTRY.md, registered before any data): arm S8K27H, S8S27H's recipe with the in-process engine releasing
only its KV cache while the trainer runs (train/grpo_q16.py --shared-base --kv-sleep, share 0.85; mlsys qrlkv27h_pod.sh), seeds 0-2,
against the FP8 offload arm on the same GPU, S8T27HO seeds 0-2 (RL-27B-H100-OFFLOAD).
  (i)   every S8K27H run completes its 100 steps
  (ii)  S8K27H's mean GPU-hours per run at most 1.00x S8T27HO's
  (iii) final accuracy (MATH-500, OlympiadBench and Minerva pooled, bf16 engine; bench/score_rl27.summary), S8K27H minus S8T27HO:
        Welch 95% interval within +-5 points, else inconclusive
Descriptive: against S8S27H (one copy without the release) and Q27 (bf16 on two A100s); step and generation time; the KV cache.
Runs trained and evaluated by 2026-10-26 12:00 UTC count.
    python3 bench/score_rlkv27h.py [--json results/std_bench/rlkv27h_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rl27 as R  # noqa: E402
from score_rlmulti import welch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ours = [R.summary(f"qwen27_dapo_S8K27H_s{k}", 1) for k in (0, 1, 2)]
    ref = [R.summary(f"qwen27_dapo_S8T27HO_s{k}", 1) for k in (0, 1, 2)]
    one = [R.summary(f"qwen27_dapo_S8S27H_s{k}", 1) for k in range(5)]
    for x in ours:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v: dict = {"steps": [x["steps"] for x in ours]}
    done = [x for x in ours if x["final_bf16"] is not None and x["steps"] >= 100]
    if len(done) == 3:
        v["i"] = {"met": all(x["steps"] >= 100 for x in ours)}
        h1, h0 = st.fmean(x["gpu_hours"] for x in ours), st.fmean(x["gpu_hours"] for x in ref)
        v["ii"] = {"S8K27H": h1, "S8T27HO": h0, "ratio": h1 / h0, "met": h1 <= h0}
        v["iii"] = welch([x["final_bf16"] for x in ours], [x["final_bf16"] for x in ref])
        so = [x["final_bf16"] for x in one if x["final_bf16"] is not None]
        v["descriptive"] = {"minus_S8S27H": welch([x["final_bf16"] for x in ours], so) if len(so) >= 2 else None,
                            "t_mean": st.fmean(x["t_mean"] for x in ours), "t_gen_mean": st.fmean(x["t_gen_mean"] for x in ours)}
    else:
        v["status"] = f"{len(done)} of 3 S8K27H runs complete and evaluated"
    print("verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": ours, "verdict": v}, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
