"""RL-H100-FAST (docs/REGISTRY.md, registered 2026-10-18): Qwen3.5-9B math RL on one H100 with fast trainer kernels in both
arms. S8TF = S8T (online FP8 engine with the plugin at 0.35, weight-matched trainer, TIS 2) with flash-linear-attention and
train/grpo_q4.py --fp8-train w16; S0TF = S0T (bf16 engine asleep during training at 0.65, bf16 trainer) with
flash-linear-attention; seeds 0-2 each, 200 steps.
  (i)   every run of both arms completes its 200 steps
  (ii)  S8TF's mean GPU-hours per run at most 1.1x S0TF's
  (iii) mean final accuracy on OlympiadBench+Minerva pooled (each on its own engine), S8TF minus S0TF: Welch 95% interval
        within +-5 points (else inconclusive)
  (iv)  no S8TF run ends below its pre-RL accuracy
Descriptive: generation and trainer time per step, the GPU-hours ratio, mismatch, TIS weights, against S8T, S0T and S0G.
    python3 bench/score_rlfasth.py [--json results/std_bench/rlfasth_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

Q.EV["S8TF"] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R, steps = {"S8TF": [], "S0TF": []}, {}
    for arm in R:
        for s in (0, 1, 2):
            rs = Q.rows(arm, s)
            steps[f"{arm}_s{s}"] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            if x is None or x.get("hard_final") is None:
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            x["trainer_s"] = st.fmean(r["t"] - r["t_gen"] for r in rs)
            R[arm].append(x)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {"steps": steps}
    if len(R["S8TF"]) == 3 and len(R["S0TF"]) == 3:
        f, b = R["S8TF"], R["S0TF"]
        v["i"] = {"met": all(n >= 200 for n in steps.values())}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["ii"] = {"S8TF": h1, "S0TF": h0, "ratio": h1 / h0, "met": h1 <= 1.1 * h0}
        v["iii"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
        below = [x["seed"] for x in f if not x["hard_final"] > x["hard_base"]]
        v["iv"] = {"below": below, "met": not below}
        v["trainer_s"] = {"S8TF": st.fmean(x["trainer_s"] for x in f), "S0TF": st.fmean(x["trainer_s"] for x in b)}
        v["gen_s"] = {"S8TF": st.fmean(x["t_gen_mean"] for x in f), "S0TF": st.fmean(x["t_gen_mean"] for x in b)}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
