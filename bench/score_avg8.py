"""EVAL-BF16-AVG8 (docs/REGISTRY.md, registered before any data): RL-H100-POWER's final policies (S0TF, S8TF, S4TF, S8MH; seeds 0-5)
re-scored on ONE common engine, vLLM bf16 with the exact adapter, with 8 samples per problem (temperature 0.6, top-p 0.95, at most
2,048 tokens, non-thinking) on OlympiadBench+Minerva pooled (avg@8; bench/std_bench4.py -> results/std_bench_avg8/<run>/bf16_exact).
The v5d mock review (R2, M5): own-engine greedy scoring conflates training with inference precision, and greedy scoring of 947
problems has a standard error near the spread between seeds.
  (i)   S8TF minus S0TF: Welch 95% interval within +-3 points (equivalent)
  (ii)  S8MH minus S0TF: the same
Descriptive: S4TF minus S0TF; per arm, the mean of avg@8 on the bf16 engine against the own-engine greedy mean (same seeds); the
base model's avg@8 (results/std_bench_avg8/base/bf16_base).
Stopping rule: the adapters scored by 2026-10-24 00:00 UTC, at least five seeds per arm (else incomplete).
    python3 bench/score_avg8.py [--json results/std_bench/avg8_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

R8 = LB / "results/std_bench_avg8"
ARMS = ("S0TF", "S8TF", "S4TF", "S8MH")
BENCHES = ("olympiad", "minerva")
STOP = dt.datetime(2026, 10, 24, 0, 0, tzinfo=dt.timezone.utc)
for _a in ("S8TF", "S4TF", "S8MH"):
    Q.EV[_a] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")


def avg8(run: str, variant: str = "bf16_exact") -> float | None:
    vals = []
    for b in BENCHES:
        f = R8 / run / variant / f"{b}.jsonl"
        if not f.exists() or dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc) > STOP:
            return None
        vals += [float(json.loads(x)["correct"]) for x in f.read_text().splitlines() if x.strip()]
    return 100 * sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    s8 = {arm: {k: avg8(f"qwen9_dapo_{arm}_s{k}") for k in range(6)} for arm in ARMS}
    s8 = {arm: {k: x for k, x in d.items() if x is not None} for arm, d in s8.items()}
    greedy = {arm: {k: (Q.summary(arm, k) or {}).get("hard_final") for k in d} for arm, d in s8.items()}
    out = {"avg8": s8, "greedy_own_engine": greedy, "base_avg8": avg8("base", "bf16_base")}
    v = {"seeds": {arm: sorted(d) for arm, d in s8.items()}}
    if all(len(d) >= 6 for d in s8.values()) or (dt.datetime.now(dt.timezone.utc) >= STOP and all(len(d) >= 5 for d in s8.values())):
        ref = list(s8["S0TF"].values())
        for key, arm in (("i", "S8TF"), ("ii", "S8MH"), ("descriptive_S4TF", "S4TF")):
            w = welch(list(s8[arm].values()), ref)
            if key != "descriptive_S4TF":
                w["met"] = -3.0 <= w["ci95"][0] and w["ci95"][1] <= 3.0
            v[key] = w
        v["avg8_vs_greedy"] = {arm: {"avg8": st.fmean(s8[arm].values()),
                                     "greedy": st.fmean(x for x in greedy[arm].values() if x is not None)} for arm in ARMS}
    else:
        v["status"] = "incomplete"
    out["verdict"] = v
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
