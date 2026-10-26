"""RL-LONG-PROBE (docs/REGISTRY.md, registered before any data; a systems probe): where the trainer's copy of the base lives, at
long rollouts (thinking mode, at most 8,192 new tokens, 12 steps, seed 0; mlsys jobs/pod_runs/rlprobe_pod.sh). Per arm, from its
log (<adapters>/probe8k/probe8k_<ARM>_s0/log.jsonl, bench/runlog.py) and its run log (lowbit logs/rlprobe/): steps completed, mean
seconds per step and per step generating over steps 3-12, generated tokens per step and per second of generation, mean response
length, the engine's KV cache. For each model (9B on one L40S: O9L, T9L, S9L, K9L; 27B on one H100: O27, T27, S27, K27):
  (i)   the KV-releasing copy's (K) mean step time at most 1.00x the offloading arm's (O)
  (ii)  the plain shared copy's (S) mean generation time per step at least 1.2x K's
Descriptive: two copies (T): whether they fit, and their times. A prediction whose arms did not complete their steps is not scored.
    python3 bench/score_rlprobe.py [--json results/rlprobe_verdict.json]"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import runlog  # noqa: E402

AD = Path("$SCRATCH/std_bench/adapters/probe8k")
STEPS, FIRST = 12, 3
MODELS = {"9B": ("O9L", "T9L", "S9L", "K9L"), "27B": ("O27", "T27", "S27", "K27")}


def arm(code: str) -> dict:
    name = f"probe8k_{code}_s0"
    rs = runlog.read(AD / name / "log.jsonl")
    log = LB / "logs/rlprobe" / f"{name}.out"
    t = log.read_text(errors="replace") if log.exists() else ""
    kv = re.search(r"Available KV cache memory: ([0-9.]+) GiB", t)
    oom = bool(re.search(r"out of memory|OutOfMemory", t))
    out = {"steps": rs[-1]["step"] if rs else 0, "kv_gib": float(kv.group(1)) if kv else None, "oom": oom}
    use = [r for r in rs if r["step"] >= FIRST]
    if rs and rs[-1]["step"] >= STEPS and use:
        out.update(t=st.fmean(r["t"] for r in use), t_gen=st.fmean(r["t_gen"] for r in use),
                   tokens=st.fmean(r.get("tokens", 0) for r in use), len=st.fmean(r.get("len", 0) for r in use),
                   gen_tok_s=sum(r.get("tokens", 0) for r in use) / sum(r["t_gen"] for r in use))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for m, codes in MODELS.items():
        arms = {c: arm(c) for c in codes}
        O, T, S, K = (arms[c] for c in codes)
        v: dict = {}
        if "t" in K and "t" in O:
            v["i"] = {"K_over_O_step": K["t"] / O["t"], "met": K["t"] <= O["t"]}
        if "t_gen" in S and "t_gen" in K:
            v["ii"] = {"S_over_K_gen": S["t_gen"] / K["t_gen"], "met": S["t_gen"] >= 1.2 * K["t_gen"]}
        res[m] = {"arms": arms, "verdict": v}
        for c, x in arms.items():
            print(m, c, json.dumps({k: (round(y, 2) if isinstance(y, float) else y) for k, y in x.items()}))
        print(m, "verdict:", json.dumps(v, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
