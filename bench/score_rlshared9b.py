"""RL-SHARED-9B (docs/REGISTRY.md, registered before any data): one L40S, Qwen3.5-9B, 20 GRPO steps each of (a) two copies at engine
share 0.45 and (b) one copy (--shared-base) at 0.55; logs <adapters>/qwen9_dapo_SH9_{two,one}/log.jsonl.
  (i) (b)'s mean generation time per step over steps 2-20 at most 0.90x (a)'s.
    python3 bench/score_rlshared9b.py [--json results/std_bench/rlshared9b_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

AD = Path("$SCRATCH/std_bench/adapters")


def rows(name: str) -> list[dict]:
    f = AD / name / "log.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    arms = {k: [r for r in rows(f"qwen9_dapo_SH9_{k}") if 2 <= r.get("step", 0) <= 20] for k in ("two", "one")}
    v = {"steps": {k: len(x) for k, x in arms.items()}}
    if all(len(x) == 19 for x in arms.values()):
        m = {k: {"t_gen": st.fmean(r["t_gen"] for r in x), "t": st.fmean(r["t"] for r in x),
                 "mismatch": st.fmean(r["mismatch"] for r in x if r.get("mismatch") is not None)} for k, x in arms.items()}
        v["means"] = m
        v["i"] = {"ratio": m["one"]["t_gen"] / m["two"]["t_gen"], "met": m["one"]["t_gen"] <= 0.90 * m["two"]["t_gen"]}
        v["step_ratio"] = m["one"]["t"] / m["two"]["t"]
    else:
        v["status"] = "incomplete"
    print(json.dumps(v, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
