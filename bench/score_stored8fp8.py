"""STORED-FP8-ACC (docs/REGISTRY.md; this scorer written before any data): FP8-stored adapters (lowbit_lora.adapters quantize
--fmt fp8: E4M3 codes with one float32 scale per row, served as LoRA requests) against the exact adapters, on vLLM's online FP8
engine with the plugin, for the nine Qwen3.5-9B RL adapters of STORED-FP8BASE and -2 (math S0G, code, tool-integrated math;
seeds 0-2), exact and stored in one engine start per adapter.
  (i)  per adapter, the paired 95% interval of FP8-stored minus exact within +-2 points (math, tool-integrated math) or +-5
       points (code: LiveCodeBench, 342 problems), the bounds of STORED-FP8BASE and -2
  (ii) the mean of the nine differences within +-1 point
Descriptive: each adapter's difference and interval; this run's exact accuracy against STORED-FP8BASE(-2)'s exact run.
Files: results/std_bench/qwen9_<kind>_s<k>_stored8fp8/fp8eng_{exact,stored}/ (math/TIR: <bench>.jsonl; code: lcb_regrade.jsonl).
    python3 bench/score_stored8fp8.py [--json results/std_bench/stored8fp8_verdict.json]"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_stored as ss  # noqa: E402
import score_stored2 as s2  # noqa: E402

KINDS = (("math", "dapo_S0G", 2.0), ("code", "code", 5.0), ("tir", "tir", 2.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for kind, name, bound in KINDS:
        load = s2.load_code if kind == "code" else ss.load
        for k in (0, 1, 2):
            d = ss.RES / f"qwen9_{name}_s{k}_stored8fp8"
            ex, sto = load(d / "fp8eng_exact"), load(d / "fp8eng_stored")
            key = f"qwen9_{kind}_s{k}"
            if ex is None or sto is None:
                res[key] = {"status": "incomplete"}
                continue
            diff, lo, hi = ss.paired_diff(sto, ex)
            e = {"n": len(set(ex) & set(sto)), "exact_acc": 100 * sum(ex.values()) / len(ex),
                 "stored_acc": 100 * sum(sto.values()) / len(sto), "stored_minus_exact": [diff, lo, hi], "bound": bound,
                 "i_met": bool(lo >= -bound and hi <= bound)}
            prev = load(ss.RES / f"qwen9_{name}_s{k}_stored8" / "fp8eng_exact")
            if prev is not None:
                e["exact_minus_previous_exact"] = list(ss.paired_diff(ex, prev))
            res[key] = e
    done = [e for e in res.values() if "stored_minus_exact" in e]
    v: dict = {"n": len(done)}
    if len(done) < 9:
        v["i"] = v["ii"] = "incomplete"
    else:
        bad = [k for k, e in res.items() if not e["i_met"]]
        v["i"] = "met" if not bad else f"not met for {', '.join(bad)}"
        m = st.fmean(e["stored_minus_exact"][0] for e in done)
        v["mean_diff"] = m
        v["ii"] = "met" if abs(m) <= 1.0 else "not met"
    out = {"adapters": res, "verdict": v}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
