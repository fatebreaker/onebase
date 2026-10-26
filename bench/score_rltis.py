"""RL-FP8-TIS (docs/REGISTRY.md, registered 2026-10-17 09:54 UTC): arm S8T = RL-FP8-QMATH's S8G (Qwen3.5-9B, DAPO-Math, online
FP8 engine with the plugin, weight-matched trainer) plus truncated importance sampling (train/grpo_q_tis.py --tis 2),
seeds 0-3. Logs <adapters>/qwen9_dapo_S8T_s<k>/log.jsonl; evaluations results/std_bench/qwen9_dapo_S8T_s<k>/fp8eng_{base,exact}
(bench/score_rlqmath.py's readers).
  (i)   every S8T run ends above its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (ii)  Welch 95% interval of mean final OlympiadBench+Minerva accuracy, S8T - S0G (S0G's five seeds), within +-5 (else
        inconclusive)
Stopping rule: the seeds whose training and evaluation finished by 2026-10-20 12:00 UTC (at least three), listed with
their evaluation times. Descriptive: trainer-engine mismatch over steps 1-10 and 181-200, the mean importance weight and
truncated share, GPU-hours, MATH-500 and AIME24/25+AMC23 avg@16, and S8T against S8G per seed.
    python3 bench/score_rltis.py [--json results/std_bench/rltis_verdict.json]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_rlqmath as Q  # noqa: E402
from score_rlmulti import welch  # noqa: E402

Q.EV["S8T"] = ("fp8eng_base", "fp8eng_exact")
DEADLINE = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)


def eval_time(arm: str, s: int) -> dt.datetime | None:
    fs = [Q.RES / f"qwen9_dapo_{arm}_s{s}/fp8eng_exact/{b}.jsonl" for b in ("olympiad", "minerva")]
    if not all(f.exists() for f in fs):
        return None
    return max(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc) for f in fs)


def extra(arm: str, s: int) -> dict:
    rs = Q.rows(arm, s)
    return {"mismatch_last20": Q.mean([r.get("mismatch") for r in rs if r["step"] > 180]),
            "mismatch_max": max((r["mismatch"] for r in rs if r.get("mismatch") is not None), default=None),
            "tis_mean": Q.mean([r.get("tis_mean") for r in rs]), "tis_clip": Q.mean([r.get("tis_clip") for r in rs]),
            "tis_clip_last20": Q.mean([r.get("tis_clip") for r in rs if r["step"] > 180])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    t8 = []
    for s in a.seeds:
        x = Q.summary("S8T", s)
        et = eval_time("S8T", s)
        if x is None or x.get("hard_final") is None or et is None:
            print(f"S8T seed {s}: incomplete")
            continue
        x.update(extra("S8T", s), eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE)
        t8.append(x)
    s0 = [x for x in (Q.summary("S0G", s) for s in range(5)) if x and x.get("hard_final") is not None]
    s8 = {s: x for s, x in ((s, Q.summary("S8G", s)) for s in range(5)) if x}
    for s, x in s8.items():
        x.update(extra("S8G", s))
    for x in t8 + list(s8.values()):
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    use = [x for x in t8 if x["in_rule"]]
    v = {"seeds_used": [x["seed"] for x in use]}
    if len(use) < 3:
        v["status"] = "incomplete (fewer than three S8T seeds by the stopping date)"
    else:
        below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
        v["i"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                  "verdict": "MET" if not below else "NOT MET"}
        if len(s0) == 5:
            v["ii"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in s0])
        else:
            v["ii"] = {"status": f"S0G has {len(s0)} of 5 seeds"}
        v["S8T_vs_S8G_per_seed"] = {x["seed"]: {"S8T": x["hard_final"], "S8G": s8[x["seed"]]["hard_final"]}
                                    for x in use if x["seed"] in s8 and s8[x["seed"]].get("hard_final") is not None}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S8T": t8, "S0G": s0, "S8G": list(s8.values()), "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
