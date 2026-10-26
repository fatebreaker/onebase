"""EVAL-AVG8-ONECOPY (docs/REGISTRY.md, registered before any data): the L40S one-copy arms and their references re-scored on one
common bf16 engine with avg@8 (EVAL-BF16-AVG8's protocol: results/std_bench_avg8/<run>/bf16_exact/{olympiad,minerva}.jsonl).
  (i)   S8S9L minus S8TLF, (ii) S8MS9L minus S8ML, (iii) S8K9L minus S8OL: Welch 95% interval over seeds 0-2 within +-3 points
Descriptive: S8MS9L over every scored seed (0-4); each arm's mean. Policies scored by 2026-10-25 12:00 UTC count.
    python3 bench/score_avg8_onecopy.py [--json results/std_bench/avg8_onecopy_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_rlmulti import welch  # noqa: E402

RES = LB / "results/std_bench_avg8_gh200"   # placement amendment -GH200 (docs/REGISTRY.md): every policy on the GH200's bf16 engine
STOP = dt.datetime(2026, 10, 25, 12, 0, tzinfo=dt.timezone.utc)
PAIRS = (("i", "S8S9L", "S8TLF"), ("ii", "S8MS9L", "S8ML"), ("iii", "S8K9L", "S8OL"))


def score(arm: str, s: int) -> float | None:
    vals, files = [], [RES / f"qwen9_dapo_{arm}_s{s}" / "bf16_exact" / f"{b}.jsonl" for b in ("olympiad", "minerva")]
    if not all(f.exists() for f in files):
        return None
    if max(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc) for f in files) > STOP:
        return None
    for f in files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            c = r.get("correct")
            vals += c if isinstance(c, list) else [c]          # avg@8: the 8 samples' correctness per problem
    return 100 * sum(float(v) for v in vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--dir", default="", help="another results directory (e.g. results/std_bench_avg8, the L40S outputs)")
    a = ap.parse_args()
    if a.dir:
        global RES
        RES = LB / a.dir
    out: dict = {}
    for key, arm, ref in PAIRS:
        xs = [x for x in (score(arm, s) for s in (0, 1, 2)) if x is not None]
        rs = [x for x in (score(ref, s) for s in (0, 1, 2)) if x is not None]
        v = {"n": [len(xs), len(rs)]}
        if len(xs) == 3 and len(rs) == 3:
            w = welch(xs, rs)
            v.update(w, met=w["ci95"][0] >= -3 and w["ci95"][1] <= 3)
        out[key] = v
        print(key, arm, "minus", ref, json.dumps(v, default=str))
    ms = [x for x in (score("S8MS9L", s) for s in range(5)) if x is not None]
    out["descriptive_S8MS9L_all"] = {"n": len(ms), "mean": sum(ms) / len(ms) if ms else None}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
