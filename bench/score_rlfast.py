"""RL-L40S-FAST (docs/REGISTRY.md, registered 2026-10-18): arm S8TLF = RL-L40S's S8TL (Qwen3.5-9B, DAPO-Math, one 48 GB L40S,
vLLM's online FP8 engine with the plugin at 0.45, weight-matched trainer, TIS 2, micro-batch 1, at most 128 sequences,
expandable segments, 200 steps) with the RL-KERNELS trainer: flash-linear-attention for the gated-delta-rule layers and our
FP8 trainer kernels (train/grpo_q4.py --fp8-train fused), seeds 0-2, against S8TL seeds 0-2.
  (i)   every S8TLF run completes its 200 steps
  (ii)  mean trainer time per step (step time minus generation time) at most 0.5x S8TL's
  (iii) mean GPU-hours per run at most 0.8x S8TL's
  (iv)  mean final accuracy on OlympiadBench+Minerva pooled (each on its own FP8 engine), S8TLF minus S8TL: Welch 95%
        interval within +-5 points (else inconclusive)
Descriptive: generation time, trainer tokens/s, mismatch over steps 1-10 and 191-200, TIS weights, AIME/AMC.
    python3 bench/score_rlfast.py [--json results/std_bench/rlfast_verdict.json]"""
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

for arm in ("S8TL", "S8TLF"):
    Q.EV[arm] = ("fp8eng_base", "fp8eng_exact")


def extra(arm: str, s: int) -> dict:
    rs = Q.rows(arm, s)
    tr = [r["t"] - r["t_gen"] for r in rs]
    tok = sum(r.get("tokens", 0) for r in rs)
    mm = [r.get("mismatch") for r in rs]
    return {"trainer_s": st.fmean(tr) if tr else None, "trainer_tok_s": tok / sum(tr) if tr and sum(tr) else None,
            "mismatch_last20": st.fmean([m for m in mm[-20:] if m is not None]) if mm else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R, steps = {"S8TL": [], "S8TLF": []}, {}
    for arm in R:
        for s in (0, 1, 2):
            rs = Q.rows(arm, s)
            steps[f"{arm}_s{s}"] = rs[-1]["step"] if rs else 0
            x = Q.summary(arm, s)
            if x is None or x.get("hard_final") is None:
                print(f"{arm} seed {s}: incomplete (steps {steps[f'{arm}_s{s}']})")
                continue
            x.update(extra(arm, s))
            R[arm].append(x)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {"steps": steps}
    if len(R["S8TLF"]) == 3 and len(R["S8TL"]) == 3:
        f, b = R["S8TLF"], R["S8TL"]
        v["i"] = {"met": all(steps[f"S8TLF_s{s}"] >= 200 for s in (0, 1, 2))}
        t1, t0 = st.fmean(x["trainer_s"] for x in f), st.fmean(x["trainer_s"] for x in b)
        v["ii"] = {"S8TLF": t1, "S8TL": t0, "ratio": t1 / t0, "met": t1 <= 0.5 * t0}
        h1, h0 = st.fmean(x["gpu_hours"] for x in f), st.fmean(x["gpu_hours"] for x in b)
        v["iii"] = {"S8TLF": h1, "S8TL": h0, "ratio": h1 / h0, "met": h1 <= 0.8 * h0}
        v["iv"] = welch([x["hard_final"] for x in f], [x["hard_final"] for x in b])
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
