"""ICML-E2-PILOT (docs/REGISTRY.md, registered 2026-10-20 03:23 UTC, before any data): LoRA RL without correction on engines that
round only the activations (S8FQ to nearest, S8FS stochastically), S8MH's recipe on one H100, seeds 0-1, from the runs' logs
(<adapters>/qwen9_dapo_<arm>_s<k>/log.jsonl) and their bf16-engine evaluations (results/std_bench/<run>/bf16_{base,exact}).
  (i)   every run completes its 200 steps
  (ii)  S8FQ: mean mismatch over steps 181-200 at least 0.015 and at least 1.2x its mean over steps 1-10 (drift)
  (iii) S8FS: mean mismatch over steps 181-200 at most 1.1x its mean over steps 1-10 (no drift)
Descriptive: OlympiadBench+Minerva accuracy on the bf16 engine against S0TF and S8MH, the top-32 TV/KL diagnostics early and
late, GPU-hours.
    python3 bench/score_e2pilot.py [--json results/std_bench/e2pilot_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
import runlog  # noqa: E402

LB = Path(__file__).resolve().parents[1]
AD = Path("$SCRATCH/std_bench/adapters")
RES = LB / "results/std_bench"


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return runlog.read(f)                                     # one row per step (bench/runlog.py)


def acc(name: str, variant: str) -> float | None:
    n = c = 0
    for b in ("olympiad", "minerva"):
        f = RES / name / variant / f"{b}.jsonl"
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "correct" in r:
                n += 1
                c += float(r["correct"])
    return 100 * c / n


def summary(arm: str, k: int) -> dict:
    name = f"qwen9_dapo_{arm}_s{k}"
    rs = rows(name)
    m = lambda key, lo, hi: (st.fmean([r[key] for r in rs if lo <= r["step"] <= hi and r.get(key) is not None])  # noqa: E731
                             if any(lo <= r["step"] <= hi and r.get(key) is not None for r in rs) else None)
    out = {"run": name, "steps": rs[-1]["step"] if rs else 0, "done": (AD / name / "adapter_model.safetensors").exists()}
    if rs:
        out.update(mismatch_first10=m("mismatch", 1, 10), mismatch_last20=m("mismatch", 181, 200),
                   tv_first10=m("tv", 1, 10), tv_last20=m("tv", 181, 200), kl_first10=m("kl", 1, 10), kl_last20=m("kl", 181, 200),
                   gpu_hours=sum(r["t"] for r in rs) / 3600)
    out.update(base=acc(name, "bf16_base"), final=acc(name, "bf16_exact"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    R = {arm: [summary(arm, k) for k in (0, 1)] for arm in ("S8FQ", "S8FS")}
    for arm, xs in R.items():
        for x in xs:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    v = {}
    complete = [x for xs in R.values() for x in xs if x["done"] and x["steps"] >= 200]
    if len(complete) == 4:
        v["i"] = {"met": True}
        q = R["S8FQ"]
        v["ii"] = {"late": [x["mismatch_last20"] for x in q], "ratio": [x["mismatch_last20"] / x["mismatch_first10"] for x in q],
                   "met": all(x["mismatch_last20"] >= 0.015 and x["mismatch_last20"] >= 1.2 * x["mismatch_first10"] for x in q)}
        s = R["S8FS"]
        v["iii"] = {"ratio": [x["mismatch_last20"] / x["mismatch_first10"] for x in s],
                    "met": all(x["mismatch_last20"] <= 1.1 * x["mismatch_first10"] for x in s)}
    else:
        v["status"] = f"incomplete ({len(complete)} of 4 runs complete)"
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"runs": R, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
