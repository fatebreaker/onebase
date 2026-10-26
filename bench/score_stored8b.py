"""STORED-FP8BASE-2 (docs/REGISTRY.md, registered 2026-10-17): STORED-STRONG-2's code (CODE-Q) and tool-integrated math (TIR-RL)
adapters of Qwen3.5-9B, seeds 0-2, on vLLM's online FP8 engine with the plugin: base, exact and NVFP4-stored adapter.
  gate  judged if the bf16 gain (STORED-STRONG-2's bf16 base and exact adapter, results/std_bench/qwen9_<kind>_s<k>_stored)
        is at least 3 points
  (i)   the stored adapter on the FP8 base keeps R >= 0.8
  (ii)  its accuracy there minus the exact adapter's: paired 95% interval within +-2 points (tool-integrated math) or +-5
        (LiveCodeBench)
Descriptive: R of the exact adapter on the FP8 engine. Files: results/std_bench/qwen9_<kind>_s<k>_stored8/fp8eng_{base,exact,
stored}/ (the base in seed 0's dir); code read from lcb_regrade.jsonl.
    python3 bench/score_stored8b.py [--json results/std_bench/stored8b_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
import score_stored as ss  # noqa: E402
import score_stored2 as s2  # noqa: E402


def judge(kind: str, k: int) -> dict | None:
    load = s2.load_code if kind == "code" else ss.load
    ref, low = ss.RES / f"qwen9_{kind}_s{k}_stored", ss.RES / f"qwen9_{kind}_s{k}_stored8"
    base_dir = ss.RES / f"qwen9_{kind}_s0_stored8" / "fp8eng_base"           # the FP8 engine's base, evaluated once per kind
    base, ex = load(ref / "bf16_base"), load(ref / "bf16_exact")
    lb, le, ls = load(base_dir), load(low / "fp8eng_exact"), load(low / "fp8eng_stored")
    if None in (base, ex, lb, le, ls):
        return None
    gain = 100 * (sum(ex.values()) / len(ex) - sum(base.values()) / len(base))
    out = {"n": len(base), "gain": gain, "judged": gain >= 3}
    for lab, v in (("exact", le), ("stored", ls)):
        r = ss.rr.retention(base, ex, lb, v, n_boot=10000)
        out[f"R_{lab}"] = {"R": r["R"], "R_ci95": r["R_ci95"], "acc": 100 * r["acc"]["rl_q"]}
    dd, lo, hi = ss.paired_diff(ls, le)
    out["stored_minus_exact"] = [dd, lo, hi]
    if out["judged"]:
        out["i"] = {"R": out["R_stored"]["R"], "met": out["R_stored"]["R"] >= 0.8}
        out["ii"] = {"diff": [dd, lo, hi], "bar": s2.BAR[kind], "met": lo >= -s2.BAR[kind] and hi <= s2.BAR[kind]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for kind in ("code", "tir"):
        for k in (0, 1, 2):
            e = judge(kind, k)
            if e is None:
                print(f"{kind} seed {k}: incomplete")
                continue
            res[f"{kind}_s{k}"] = e
            print(f"{kind}_s{k}: gain {e['gain']:+.1f} ({'judged' if e['judged'] else 'not judged'}); exact R {e['R_exact']['R']:.2f}, "
                  f"stored R {e['R_stored']['R']:.2f}; stored - exact {e['stored_minus_exact'][0]:+.1f} "
                  f"[{e['stored_minus_exact'][1]:+.1f}, {e['stored_minus_exact'][2]:+.1f}]"
                  + "".join(f"; ({p}) {'MET' if e[p]['met'] else 'NOT MET'}" for p in ("i", "ii") if p in e))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
