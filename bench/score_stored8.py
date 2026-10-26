"""STORED-FP8BASE (docs/REGISTRY.md, registered 2026-10-17 11:31 UTC): QMATH-CONV's Qwen3.5-9B math adapters (S0G seeds 0-2)
on vLLM's online FP8 engine with the plugin, exact and NVFP4-stored; MATH-500, OlympiadBench and Minerva pooled.
  (i)   the stored adapter on the FP8 base keeps R >= 0.8 of the bf16 gain (bf16 base and exact from QMATH-CONV's
        evaluation, results/std_bench/qwen9_dapo_S0G_s<k>_conv)
  (ii)  the paired 95% interval of stored minus exact on the FP8 base within +-2 points
Descriptive: R of the exact adapter on the FP8 engine, and its accuracy there against the bf16 engine's.
Files: results/std_bench/qwen9_dapo_S0G_s<k>_stored8/fp8eng_{base,exact,stored}/<bench>.jsonl (the base in seed 0's dir).
    python3 bench/score_stored8.py [--json results/std_bench/stored8_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_stored as S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    base_dir = S.RES / "qwen9_dapo_S0G_s0_stored8/fp8eng_base"      # the FP8 engine's base, evaluated once
    res = {}
    for k in (0, 1, 2):
        ref, low = S.RES / f"qwen9_dapo_S0G_s{k}_conv", S.RES / f"qwen9_dapo_S0G_s{k}_stored8"
        if not (low / "fp8eng_base").exists() and base_dir.exists():
            (low / "fp8eng_base").symlink_to(base_dir)               # the same base for every adapter
        e = S.judge(ref, low, "fp8eng_base", "fp8eng_exact", "fp8eng_stored", {})
        if e is None:
            print(f"S0G seed {k}: incomplete")
            continue
        ex8 = S.load(low / "fp8eng_exact")
        ex16 = S.load(ref / "bf16_exact")
        d, lo, hi = S.paired_diff(ex8, ex16)
        e["exact_fp8_minus_bf16"] = [d, lo, hi]
        res[f"qwen9_math_s{k}"] = e
        print(json.dumps({"adapter": f"S0G seed {k}", **{x: e.get(x) for x in ("gain", "R_exact", "R_stored", "stored_minus_exact",
                                                                                  "exact_fp8_minus_bf16", "i", "ii")}}))
    v = {"n": len(res), "i_met": sum(bool(e.get("i", {}).get("met")) for e in res.values()),
         "ii_met": sum(bool(e.get("ii", {}).get("met")) for e in res.values()),
         "judged": sum(bool(e.get("judged")) for e in res.values())}
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"adapters": res, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
