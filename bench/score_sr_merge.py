"""SR-MERGE (docs/REGISTRY.md, registered 2026-10-13 15:49 UTC, before any data): the registered rule on
results/sr_merge/a<k>_logits.json (jobs/sr_merge_logits.sbatch or jobs/sr_merge_pod.sh; mlsys.fidelity output:
logit_gain[variant] = {gain, gain_ci95, corr, ...}).
  valid  the unchanged copy's g in [0.98, 1.02]
  (i)    unbiased: each SR merge (bf16_sr_merge, fp8_srfrozen, nvfp4_srfrozen) has g in [0.7, 1.3] on at least 6 of the 8
         adapters (75% of the valid ones if fewer are valid), per format
  (ii)   noise: corr(bf16 SR) >= corr(FP8 SR) >= corr(NVFP4 SR) on at least 6 of 8 (75% of the valid ones)
  (iii)  nearest: nvfp4_frozen g <= 0.05 on every valid adapter, and fp8_frozen's g between nvfp4_frozen's and
         bf16_merge's on at least 6 of 8 (75% of the valid ones)
    python bench/score_sr_merge.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results/sr_merge"
SR = ("bf16_sr_merge:rl", "fp8_srfrozen:rl", "nvfp4_srfrozen:rl")
NEAR = ("bf16_merge:rl", "fp8_frozen:rl", "nvfp4_frozen:rl")


def load(k: int) -> dict | None:
    f = RES / f"a{k}_logits.json"
    if not f.exists():
        return None
    lg = json.loads(f.read_text()).get("logit_gain", {})
    ctl = next((v for key, v in lg.items() if key.startswith("lora=") and key.endswith("/copy")), None)
    return {"control": ctl and ctl["gain"], **{v: lg.get(v) for v in SR + NEAR}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = {k: load(k) for k in range(1, 9)}
    f = lambda d, key: "--" if not d or not d.get(key) else f"{d[key]['gain']:.3f}/{d[key].get('corr', float('nan')):.2f}"  # noqa: E731
    print(f"{'a':>2s} {'control':>8s} " + " ".join(f"{v.split(':')[0]:>16s}" for v in NEAR + SR) + "   (g/corr)")
    for k, d in rows.items():
        ctl = "--" if not d or d["control"] is None else f"{d['control']:.3f}"
        print(f"{k:>2d} {ctl:>8s} " + " ".join(f"{f(d, v):>16s}" for v in NEAR + SR))
    missing = [k for k, d in rows.items() if d is None or any(d.get(v) is None for v in SR + NEAR)]
    valid = [k for k, d in rows.items() if k not in missing and d["control"] is not None and 0.98 <= d["control"] <= 1.02]
    if missing:
        print(f"incomplete adapters: {missing}; no verdict until all eight are in")
        return 1
    need = 6 if len(valid) == 8 else math.ceil(0.75 * len(valid))
    v = {}
    for sr in SR:
        ok = [k for k in valid if 0.7 <= rows[k][sr]["gain"] <= 1.3]
        v[f"i {sr}"] = {"met": len(ok) >= need, "n": len(ok)}
    order = [k for k in valid if rows[k]["bf16_sr_merge:rl"]["corr"] >= rows[k]["fp8_srfrozen:rl"]["corr"]
             >= rows[k]["nvfp4_srfrozen:rl"]["corr"]]
    v["ii"] = {"met": len(order) >= need, "n": len(order)}
    nv = [k for k in valid if rows[k]["nvfp4_frozen:rl"]["gain"] <= 0.05]
    between = [k for k in valid if min(rows[k]["nvfp4_frozen:rl"]["gain"], rows[k]["bf16_merge:rl"]["gain"])
               <= rows[k]["fp8_frozen:rl"]["gain"] <= max(rows[k]["nvfp4_frozen:rl"]["gain"], rows[k]["bf16_merge:rl"]["gain"])]
    v["iii"] = {"met": len(nv) == len(valid) and len(between) >= need, "nvfp4_le_0.05": len(nv), "fp8_between": len(between)}
    print(f"valid adapters: {valid} (need {need})")
    for key, r in v.items():
        print(f"  ({key}) {'MET' if r['met'] else 'NOT MET'}  {r}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "valid": valid, "verdict": v}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
