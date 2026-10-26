"""STORED-27B (docs/REGISTRY.md): NVFP4-stored adapters of Q27-RL's Qwen3.8-27B math adapters on NVIDIA's NVFP4 release,
judged as STORED-STRONG (bench/score_stored.judge): the bf16 gain from the original's base/exact
(results/std_bench/qwen27_dapo_s<k>_stored), the release's base/exact/stored (qwen27_dapo_s<k>_stored_nvrel).
    python3 bench/score_stored27.py [--json results/std_bench/stored27_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, str(LB / "bench"))
import score_stored as ss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for k in (0, 1, 2):
        e = ss.judge(RES / f"qwen27_dapo_s{k}_stored", RES / f"qwen27_dapo_s{k}_stored_nvrel", "bf16_base", "bf16_exact",
                     "bf16_stored", {})
        if e:
            res[f"qwen27_math_s{k}"] = e
    for k, e in res.items():
        print(f"{k}: gain {e['gain']:+.1f} ({'judged' if e['judged'] else 'not judged'}); "
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
