"""RL-FLASHRL-LORA (docs/REGISTRY.md, registered 2026-10-19 before any data; the v3 mock review, R2 M5 and R3): FlashRL's recipe
for LoRA on the lent H100: arm S4TF = the W8A8 FP8 engine with our plugin at 0.35, a bf16 trainer (not weight-matched) with
flash-linear-attention, truncated importance sampling C = 2 (train/grpo_q4.py --engine fp8 --tis 2), RL-FP8-QMATH's recipe,
seeds 0-2, against RL-H100-FAST's S0TF (tuned bf16 baseline) and S8TF (weight-matched trainer, TIS). Logs
<adapters>/qwen9_dapo_S4TF_s<k>/log.jsonl; evaluations results/std_bench/qwen9_dapo_S4TF_s<k>/fp8eng_{base,exact}.
  (i)   every S4TF run completes its 200 steps
  (ii)  no S4TF run ends at or below its own pre-RL accuracy on OlympiadBench+Minerva pooled
  (iii) Welch 95% interval of mean final OlympiadBench+Minerva accuracy, S4TF - S0TF (three seeds each), within +-5 (else
        inconclusive)
Stopping rule: the seeds whose training and evaluation finished by 2026-10-22 12:00 UTC (at least two). Descriptive: the
mismatch over steps 1-10 and 181-200 and the TIS clip rate against S8TF's, GPU-hours, trainer peak memory, S4TF - S8TF.
    python3 bench/score_rlflash.py [--json results/std_bench/rlflash_verdict.json]"""
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
import score_rltis as T  # noqa: E402
from score_rlmulti import welch  # noqa: E402

for _a in ("S4TF", "S8TF"):
    Q.EV[_a] = ("fp8eng_base", "fp8eng_exact")
Q.EV["S0TF"] = ("bf16_base", "bf16_exact")
DEADLINE = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.timezone.utc)


def late(rs: list[dict], key: str, lo: int, hi: int) -> float | None:
    v = [r[key] for r in rs if lo <= r["step"] <= hi and r.get(key) is not None]
    return st.fmean(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    runs, steps = [], {}
    for s in (0, 1, 2):
        rs = Q.rows("S4TF", s)
        steps[s] = rs[-1]["step"] if rs else 0
        x = Q.summary("S4TF", s)
        et = T.eval_time("S4TF", s)
        if x is None or x.get("hard_final") is None or et is None:
            print(f"S4TF seed {s}: incomplete (steps {steps[s]})")
            continue
        x.update(eval_utc=et.strftime("%Y-%m-%dT%H:%M"), in_rule=et <= DEADLINE,
                 mismatch_first10=late(rs, "mismatch", 1, 10), mismatch_last20=late(rs, "mismatch", 181, 200),
                 tis_clip_last20=late(rs, "tis_clip", 181, 200), trainer_s=st.fmean(r["t"] - r["t_gen"] for r in rs))
        runs.append(x)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}))
    ref = {arm: [x for x in (Q.summary(arm, k) for k in range(3)) if x and x.get("hard_final") is not None]
           for arm in ("S0TF", "S8TF")}
    use = [x for x in runs if x["in_rule"]]
    v = {"steps": steps, "seeds_used": [x["seed"] for x in use]}
    if len(use) < 2:
        v["status"] = "incomplete (fewer than two S4TF seeds by the stopping date)"
    else:
        v["i"] = {"met": all(steps[x["seed"]] >= 200 for x in use)}
        below = [x["seed"] for x in use if not x["hard_final"] > x["hard_base"]]
        v["ii"] = {"final_minus_base": {x["seed"]: x["hard_final"] - x["hard_base"] for x in use}, "below": below,
                   "met": not below}
        v["iii"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in ref["S0TF"]]) \
            if len(ref["S0TF"]) == 3 else {"status": f"S0TF has {len(ref['S0TF'])} of 3 seeds"}
        if len(ref["S8TF"]) == 3:
            v["vs_S8TF"] = welch([x["hard_final"] for x in use], [x["hard_final"] for x in ref["S8TF"]])
        v["gpu_hours"] = {"S4TF": st.fmean(x["gpu_hours"] for x in use),
                          **{arm: st.fmean(x["gpu_hours"] for x in xs) for arm, xs in ref.items() if xs}}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"S4TF": runs, **ref, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
