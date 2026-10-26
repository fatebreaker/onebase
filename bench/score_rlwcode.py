"""RL-WONLY-CODE (docs/REGISTRY.md, registered 2026-10-18 22:27 UTC, before any data): arm S8WC, CODE-Q's recipe on one A100-80GB
with vLLM's online FP8 engine (weight-only FP8 Marlin there), the weight-matched trainer with flash-linear-attention, no
importance correction (train/grpo_code3.py --engine fp8 --match fp8w), seeds 0-2, against CODE-Q's three adapters on the same
A100 bf16 engine: LiveCodeBench v5+v6 pass@1 (342 problems), regraded by bench/lcb_regrade.py.
  (i)   every S8WC run completes its 100 steps
  (ii)  every run's mean mismatch over steps 81-100 is at most 1.5x the mean of CODE-Q's three runs over the same steps
  (iii) every run ends above the base's pass@1 on the A100 bf16 engine
  (iv)  Welch 95% interval of mean final pass@1, S8WC minus CODE-Q (same engine), within +-5 (else inconclusive)
Stopping rule: the seeds whose training and evaluation finished by 2026-10-21 00:00 UTC (at least two).
Files: logs <adapters>/qwen9_code_S8WC_s<k>/log.jsonl; evaluations results/std_bench/qwen9_code_S8WC_s<k>/{bf16_exact,fp8eng_exact};
CODE-Q's A100 evaluations results/std_bench/qwen9_code_s<k>/{bf16_base,bf16_exact}.
    python3 bench/score_rlwcode.py [--json results/std_bench/rlwcode_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
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

DEADLINE = dt.datetime(2026, 10, 21, 0, 0, tzinfo=dt.timezone.utc)


def acc(d: Path) -> float | None:
    v = load_code(d)
    return 100 * sum(v.values()) / len(v) if v else None


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def late(rs: list[dict], lo: int = 81, hi: int = 100) -> float | None:
    v = [r["mismatch"] for r in rs if lo <= r["step"] <= hi and r.get("mismatch") is not None]
    return st.fmean(v) if v else None


def eval_time(name: str) -> dt.datetime | None:
    f = RES / name / "bf16_exact"
    fs = [x for x in f.glob("*.jsonl")] if f.exists() else []
    return max(dt.datetime.fromtimestamp(x.stat().st_mtime, dt.timezone.utc) for x in fs) if fs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    ours = []
    for k in (0, 1, 2):
        name = f"qwen9_code_S8WC_s{k}"
        rs = rows(name)
        et = eval_time(name)
        x = {"seed": k, "steps": rs[-1]["step"] if rs else 0, "gpu_hours": sum(r["t"] for r in rs) / 3600 if rs else None,
             "t_gen_mean": st.fmean(r["t_gen"] for r in rs) if rs else None,
             "trainer_s": st.fmean(r["t"] - r["t_gen"] for r in rs) if rs else None,
             "mismatch_first10": late(rs, 1, 10), "mismatch_late": late(rs), "final": acc(RES / name / "bf16_exact"),
             "final_own": acc(RES / name / "fp8eng_exact"), "eval_utc": et.strftime("%Y-%m-%dT%H:%M") if et else None,
             "in_rule": bool(et and et <= DEADLINE)}
        ours.append(x)
        print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()}))
    ref = []
    for k in (0, 1, 2):
        name = f"qwen9_code_s{k}"
        rs = rows(name)
        ref.append({"seed": k, "final": acc(RES / name / "bf16_exact"), "mismatch_late": late(rs),
                    "gpu_hours": sum(r["t"] for r in rs) / 3600 if rs else None})
    base = acc(RES / "qwen9_code_s0" / "bf16_base")
    ref_late = st.fmean([x["mismatch_late"] for x in ref if x["mismatch_late"] is not None])
    print(json.dumps({"CODEQ": ref, "base_a100_bf16": base, "codeq_late_mismatch": ref_late}))
    use = [x for x in ours if x["in_rule"] and x["final"] is not None]
    v = {"steps": [x["steps"] for x in ours], "seeds_used": [x["seed"] for x in use]}
    if len(use) < 2 or base is None or any(x["final"] is None for x in ref):
        v["status"] = "incomplete"
    else:
        v["i"] = {"met": all(x["steps"] >= 100 for x in use)}
        v["ii"] = {"late": {x["seed"]: x["mismatch_late"] for x in use}, "codeq": ref_late,
                   "met": all(x["mismatch_late"] is not None and x["mismatch_late"] <= 1.5 * ref_late for x in use)}
        below = [x["seed"] for x in use if not x["final"] > base]
        v["iii"] = {"base": base, "below": below, "met": not below}
        v["iv"] = welch([x["final"] for x in use], [x["final"] for x in ref])
    print("verdict:", json.dumps(v))
    # extension (registered 2026-10-19 16:32 UTC, descriptive): every completed seed, S8WC up to six and CODE-Q four
    xo = [x for x in ours] + [{"seed": k, "final": acc(RES / f"qwen9_code_S8WC_s{k}" / "bf16_exact"),
                               "final_own": acc(RES / f"qwen9_code_S8WC_s{k}" / "fp8eng_exact")} for k in (3, 4, 5)]
    xr = ref + [{"seed": k, "final": acc(RES / f"qwen9_code_s{k}" / "bf16_exact")} for k in (3, 4, 5)]   # extension 2: seeds 4-5
    xs, rs_ = [x["final"] for x in xo if x["final"] is not None], [x["final"] for x in xr if x["final"] is not None]
    ext = {"S8WC": xs, "CODEQ": rs_, "welch": welch(xs, rs_) if len(xs) >= 2 and len(rs_) >= 2 else None}
    print("extension:", json.dumps(ext))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8WC": ours, "CODEQ": ref, "base": base, "verdict": v, "extension": ext}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
