"""STORED-STRONG-2 (docs/REGISTRY.md): NVFP4-stored adapters of CODE-Q's and TIR-RL's Qwen3.5-9B adapters (seeds 0-2) on
our NVFP4 base, every variant of an adapter in one session:
  gate  judged if the bf16 gain (exact adapter minus base, bf16) is at least 3 points
  (i)   the NVFP4-stored adapter on our NVFP4 base keeps R >= 0.8 (R against that base, paired bootstrap interval)
  (ii)  its accuracy there minus the exact adapter's on that base: paired 95% interval within +-2 points (tool-integrated
        math, 1,447 problems) or +-5 points (LiveCodeBench, 342 problems)
Descriptive: R of the exact adapter on the NVFP4 base.
Files: results/std_bench/qwen9_{code,tir}_s<k>_stored/<variant>/: code lcb_regrade.jsonl (bench/lcb_regrade.py), math
math500/olympiad/minerva.jsonl.
    python3 bench/score_stored2.py [--json results/std_bench/stored2_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, str(LB / "bench"))
import score_stored as ss  # noqa: E402

BAR = {"code": 5.0, "tir": 2.0}


def load_code(d: Path) -> dict | None:
    f = d / "lcb_regrade.jsonl"
    if not f.exists():
        return None
    return {f"lcb:{r['id']}": int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())}


def judge(d: Path, kind: str) -> dict | None:
    load = load_code if kind == "code" else ss.load
    base, ex, lb, le, ls = (load(d / v) for v in ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_stored"))
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
        out["ii"] = {"diff": [dd, lo, hi], "bar": BAR[kind], "met": lo >= -BAR[kind] and hi <= BAR[kind]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for kind in ("code", "tir"):
        for k in (0, 1, 2):
            e = judge(RES / f"qwen9_{kind}_s{k}_stored", kind)
            if e:
                res[f"{kind}_s{k}"] = e
    for k, e in res.items():
        print(f"{k}: gain {e['gain']:+.1f} ({'judged' if e['judged'] else 'not judged'}; n {e['n']}); "
              + ", ".join(f"{lab[2:]} R {v['R']:.2f} [{v['R_ci95'][0]:.2f}, {v['R_ci95'][1]:.2f}]" for lab, v in e.items()
                          if lab.startswith("R_"))
              + f"; stored - exact {e['stored_minus_exact'][0]:+.1f} [{e['stored_minus_exact'][1]:+.1f}, {e['stored_minus_exact'][2]:+.1f}]")
        for p in ("i", "ii"):
            if p in e:
                print(f"   ({p}) {'MET' if e[p]['met'] else 'NOT MET'}: {json.dumps(e[p])}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
