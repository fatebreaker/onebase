"""RL-L40S-CODE (docs/REGISTRY.md, registered 2026-10-17): arm S8TLC, CODE-Q's recipe (Qwen3.5-9B GRPO on reference-verified
PrimeIntellect problems, rank 64, lr 2e-5, 100 steps of 16 x 8, at most 2,048 new tokens) on the one-GPU FP8 system of RL-L40S
(one 48 GB L40S; vLLM's online FP8 engine with the plugin, weight-matched trainer, truncated importance sampling C = 2,
engine share 0.45, micro-batch 1, at most 128 sequences, expandable segments; train/grpo_code3.py), seeds 0-2, against
CODE-Q's three bf16 adapters (one A100-80GB each, bf16 engine at 35%), all evaluated on one L40S bf16 engine with the exact
adapter: LiveCodeBench v5+v6 pass@1 (342 problems), greedy, non-thinking, regraded by bench/lcb_regrade.py.
  (i)   every S8TLC run completes its 100 steps
  (ii)  every S8TLC run ends above the base's pass@1 on that engine
  (iii) mean final pass@1, S8TLC minus CODE-Q: Welch 95% interval within +-5 points (else inconclusive)
Descriptive: step and generation times, GPU-hours (CODE-Q ran on another GPU: reported, not compared as a speedup), mismatch,
TIS weights; each S8TLC adapter on its own FP8 engine; CODE-Q's A100 scores beside their L40S re-evaluation.
Files: logs <adapters>/qwen9_code_S8TLC_s<k>/log.jsonl; evaluations results/std_bench/qwen9_code_S8TLC_s<k>/<variant>/ (the
base in seed 0's dir) and CODE-Q's re-evaluation results/std_bench/qwen9_code_s<k>_l40s/bf16_exact/.
    python3 bench/score_rllcode.py [--json results/std_bench/rllcode_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
AD = Path("$SCRATCH/std_bench/adapters")
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402
from score_rlmulti import welch  # noqa: E402
from score_stored2 import load_code  # noqa: E402


def acc(d: Path) -> float | None:
    v = load_code(d)
    return 100 * sum(v.values()) / len(v) if v else None


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def summary(k: int) -> dict:
    name = f"qwen9_code_S8TLC_s{k}"
    rs = rows(name)
    mm = [r["mismatch"] for r in rs if r.get("mismatch") is not None]
    return {"seed": k, "steps": rs[-1]["step"] if rs else 0, "gpu_hours": sum(r["t"] for r in rs) / 3600 if rs else None,
            "t_mean": st.fmean(r["t"] for r in rs) if rs else None, "t_gen_mean": st.fmean(r["t_gen"] for r in rs) if rs else None,
            "mismatch_first10": st.fmean([r["mismatch"] for r in rs if r["step"] <= 10 and r.get("mismatch") is not None] or [0])
            if mm else None, "mismatch_last10": st.fmean(mm[-10:]) if mm else None,
            "tis_mean": st.fmean([r["tis_mean"] for r in rs if r.get("tis_mean")] or [0]) if rs else None,
            "final": acc(RES / name / "bf16_exact"), "final_own": acc(RES / name / "fp8eng_exact")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ours = [summary(k) for k in (0, 1, 2)]
    base = acc(RES / "qwen9_code_S8TLC_s0" / "bf16_base")
    base_own = acc(RES / "qwen9_code_S8TLC_s0" / "fp8eng_base")
    ref = [{"seed": k, "final": acc(RES / f"qwen9_code_s{k}_l40s" / "bf16_exact"), "final_a100": acc(RES / f"qwen9_code_s{k}" / "bf16_exact"),
            "base_a100": acc(RES / f"qwen9_code_s{k}" / "bf16_base")} for k in (0, 1, 2)]
    for x in ours + ref:
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()}))
    print(json.dumps({"base_l40s_bf16": base, "base_l40s_fp8eng": base_own}))
    v = {"steps": [x["steps"] for x in ours]}
    if all(x["final"] is not None for x in ours) and all(x["final"] is not None for x in ref) and base is not None:
        v["i"] = {"met": all(x["steps"] >= 100 for x in ours)}
        below = [x["seed"] for x in ours if not x["final"] > base]
        v["ii"] = {"base": base, "below": below, "met": not below}
        v["iii"] = welch([x["final"] for x in ours], [x["final"] for x in ref])
    else:
        v["status"] = "incomplete"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8TLC": ours, "CODEQ": ref, "base": base, "base_own": base_own, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
