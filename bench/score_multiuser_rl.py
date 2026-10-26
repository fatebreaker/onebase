"""MULTI-USER-RL (docs/REGISTRY.md, registered before any data; this scorer written before any data): K users' LoRA RL jobs on one
H100, one engine and one FP8 copy (mlsys jobs/pod_runs/multiuser_rl_pod.sh -> $SCRATCH/multiuser/
mu<K>_s<seeds>; adapters qwen9_dapo_S8MU<K>H_s<seed>). Arms: S8MU1H (one user per GPU; seeds 0-3, one run each) and S8MU2H (two
users sharing the GPU; runs mu2_s0_1 and mu2_s2_3).
  (i)   every user's run completes its 200 steps
  (ii)  GPU-hours: a two-user run's RL phase (wall time on its one GPU, summed over restarts; mean over the two runs) at most 0.75x
        two one-user runs' (2x S8MU1H's mean)
  (iii) final accuracy (OlympiadBench + Minerva pooled, each policy on its own FP8 engine), S8MU2H's four users minus S8MU1H's four
        runs: Welch 95% interval within +-5 points (else inconclusive)
Descriptive: per-user step and rollout times, the trainer-engine mismatch, the engine's KV cache. Stopping: runs trained and
evaluated by 2026-10-28 12:00 UTC.
    python3 bench/score_multiuser_rl.py [--json results/std_bench/multiuser_rl_verdict.json]"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

RUNS = Path("$SCRATCH/multiuser")
ONE = {s: f"mu1_s{s}" for s in (0, 1, 2)}   # amendment -K3 (2026-10-24 14:00 UTC, before its data): seed 3 dropped
TWO = {"mu2_s0_1": (0, 1), "mu2_s2_3": (2, 3)}
for _a in ("S8MU1H", "S8MU2H"):
    Q.EV.setdefault(_a, ("fp8eng_base", "fp8eng_exact"))


def rl_hours(d: Path) -> tuple[float | None, bool]:
    """Wall time of the RL phase(s): each rl start opens a segment, its rl end closes it; a segment cut by a restart (an engine
    start while open) ends at the last event before it. Returns (hours, last segment closed)."""
    f = d / "phases.jsonl"
    if not f.exists():
        return None, False
    segs, cur, last, closed = [], None, None, False
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["phase"] == "engine" and r["event"] == "start" and cur is not None:
            segs.append(last - cur)
            cur = None
        if r["phase"] == "rl" and r["event"] == "start":
            cur, closed = r["t"], False
        elif r["phase"] == "rl" and r["event"] == "end" and cur is not None:
            segs.append(r["t"] - cur)
            cur, closed = None, True
        last = r["t"]
    if cur is not None:
        segs.append(last - cur)
    return (sum(segs) / 3600 if segs else None), closed


def user(arm: str, s: int) -> dict:
    rs = Q.rows(arm, s)
    x = Q.summary(arm, s)
    late = [r for r in rs if r["step"] >= 3]
    rec = {"steps": rs[-1]["step"] if rs else 0}
    if late:
        rec.update({"t_step": st.fmean(r["t"] for r in late), "t_gen": st.fmean(r["t_gen"] for r in late),
                    "mismatch": st.fmean(r["mismatch"] for r in late if r.get("mismatch") is not None)})
    if x and x.get("hard_final") is not None:
        rec["hard_final"] = x["hard_final"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    one, two = {}, {}
    for s, run in ONE.items():
        h, closed = rl_hours(RUNS / run)
        one[s] = {**user("S8MU1H", s), "rl_h": h, "closed": closed}
    for run, seeds in TWO.items():
        h, closed = rl_hours(RUNS / run)
        two[run] = {"rl_h": h, "closed": closed, "users": {s: user("S8MU2H", s) for s in seeds}}
    for d in [RUNS / r for r in list(ONE.values()) + list(TWO)]:
        log = d / "server.log"
        kv = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", log.read_text(errors="replace")) if log.exists() else []
        (one.get(int(d.name.split("_s")[1])) if d.name.startswith("mu1") else two[d.name])["kv_tokens"] = int(kv[-1].replace(",", "")) if kv else None
    users2 = [u for r in two.values() for u in r["users"].values()]
    v: dict = {}
    complete = (all(r["steps"] >= 200 and r["closed"] for r in one.values()) and all(r["closed"] for r in two.values())
                and all(u["steps"] >= 200 for u in users2))
    v["i"] = {"met": complete, "steps_one": {s: r["steps"] for s, r in one.items()},
              "steps_two": {s: u["steps"] for r in two.values() for s, u in r["users"].items()}}
    if complete:
        h1, h2 = st.fmean(r["rl_h"] for r in one.values()), st.fmean(r["rl_h"] for r in two.values())
        v["ii"] = {"two_user_h": h2, "one_user_h": h1, "ratio": h2 / (2 * h1), "met": h2 <= 0.75 * 2 * h1}
        if all("hard_final" in r for r in one.values()) and all("hard_final" in u for u in users2):
            v["iii"] = welch([u["hard_final"] for u in users2], [r["hard_final"] for r in one.values()])
        v["descriptive"] = {"t_step": {"S8MU1H": st.fmean(r["t_step"] for r in one.values()), "S8MU2H": st.fmean(u["t_step"] for u in users2)},
                            "t_gen": {"S8MU1H": st.fmean(r["t_gen"] for r in one.values()), "S8MU2H": st.fmean(u["t_gen"] for u in users2)},
                            "mismatch": {"S8MU1H": st.fmean(r["mismatch"] for r in one.values()), "S8MU2H": st.fmean(u["mismatch"] for u in users2)}}
    else:
        v["status"] = "incomplete"
    out = {"one": one, "two": two, "verdict": v}
    print(json.dumps(out, indent=1, default=float))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
