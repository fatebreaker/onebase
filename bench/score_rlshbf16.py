"""Score RL-SHARED-BF16 (docs/REGISTRY.md; written before any data): one bf16 copy for the bf16 engine and trainer.

Per arm, the mean step time ("t" in log.jsonl) and generation time ("t_gen") over steps 3..STEPS, against the same steps of the
seed-0 full runs, which saw the same prompts: the FP8 one copy (S8S9L; the 27B: S8S27H) and bf16 offloading (S0OL; the 27B:
S0T27HO). Prediction (i): the faster of BS55 and BS70 takes at least 1.10x S8S9L s0's mean step time. Descriptive: each arm
against S0OL, its KV cache (the run log's "Available KV cache memory"), peak host memory (Pss, results/rlshbf16/<arm>.jsonl),
and whether it completed. An arm that did not complete its steps is reported as such; (i) needs at least one 9B arm.
    python3 bench/score_rlshbf16.py [--steps 12] -> results/rlshbf16_verdict.json"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
AD = Path("$SCRATCH/std_bench/adapters")
REF = {"BS55": ("qwen9_dapo_S8S9L_s0", "qwen9_dapo_S0OL_s0"), "BS70": ("qwen9_dapo_S8S9L_s0", "qwen9_dapo_S0OL_s0"),
       "BS27": ("qwen27_dapo_S8S27H_s0", "qwen27_dapo_S0T27HO_s0")}


def steps(path: Path, lo: int, hi: int) -> list[dict]:
    if not path.exists():
        return []
    rs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rs if lo <= r.get("step", 0) <= hi]


def mean(rs: list[dict], key: str):
    v = [r[key] for r in rs if r.get(key) is not None]
    return round(st.fmean(v), 1) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    out = {"steps": [3, a.steps], "arms": {}}
    for arm, (fp8, off) in REF.items():
        rs = steps(AD / "rlshbf16" / f"rlshbf16_{arm}_s0" / "log.jsonl", 3, a.steps)
        log = LB / "logs" / "rlshbf16" / f"rlshbf16_{arm}_s0.out"
        kv = re.findall(r"Available KV cache memory: ([0-9.]+) GiB", log.read_text(errors="ignore")) if log.exists() else []
        pss_f = LB / "results" / "rlshbf16" / f"{arm}.jsonl"
        pss = [json.loads(line)["pss_gib"] for line in pss_f.read_text().splitlines() if line.strip()] if pss_f.exists() else []
        rec = {"completed": len(rs) == a.steps - 2, "steps_logged": len(rs), "t": mean(rs, "t"), "t_gen": mean(rs, "t_gen"),
               "kv_gib": float(kv[0]) if kv else None, "peak_pss_gib": max(pss) if pss else None}
        for tag, ref in (("fp8_one_copy", fp8), ("bf16_offload", off)):
            rr = steps(AD / ref / "log.jsonl", 3, a.steps)
            rec[tag] = {"run": ref, "t": mean(rr, "t"), "t_gen": mean(rr, "t_gen")}
            if rec["t"] and rec[tag]["t"]:
                rec[tag]["ratio_t"] = round(rec["t"] / rec[tag]["t"], 3)
        out["arms"][arm] = rec
    nine = [out["arms"][k] for k in ("BS55", "BS70") if out["arms"][k]["completed"]]
    if nine:
        best = min(nine, key=lambda r: r["t"])
        r = best["fp8_one_copy"].get("ratio_t")
        out["prediction_i"] = {"ratio_best_bf16_one_copy_to_S8S9L": r, "met": r is not None and r >= 1.10}
    else:
        out["prediction_i"] = {"met": None, "note": "no 9B arm completed"}
    (LB / "results" / "rlshbf16_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
