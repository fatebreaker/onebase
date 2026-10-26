"""SR-MERGE-STD (docs/REGISTRY.md, registered 2026-10-14 00:56 UTC, before any data): the registered predictions on
results/std_bench/srmerge_mistral_<math|code>.json (mlsys.fidelity: logit_gain[variant] = {gain, corr, ...}).
  (i)   valid if the unchanged copy's g is in [0.98, 1.02]
  (ii)  large update: the nearest NVFP4 merge keeps g >= 0.8 on both adapters
  (iii) noise: on both adapters corr(stochastic NVFP4 merge) < corr(nearest NVFP4 merge)
    python bench/score_srmerge_std.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results/std_bench"
SHOW = ("bf16_merge:rl", "bf16_sr_merge:rl", "fp8_frozen:rl", "fp8_srfrozen:rl", "nvfp4_frozen:rl", "nvfp4_srfrozen:rl")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows, v = {}, {"ii": [], "iii": [], "valid": []}
    for w in ("math", "code"):
        f = RES / f"srmerge_mistral_{w}.json"
        if not f.exists():
            print(f"{w}: missing")
            continue
        lg = json.loads(f.read_text())["logit_gain"]
        ctl = next((x["gain"] for k, x in lg.items() if k.startswith("lora=") and k.endswith("/copy")), None)
        rows[w] = {"control": ctl, **{k: {"g": lg[k]["gain"], "corr": lg[k].get("corr")} for k in SHOW if k in lg}}
        print(f"{w}: control {ctl:.3f}  " + "  ".join(f"{k.split(':')[0]} {lg[k]['gain']:.3f}/{lg[k].get('corr', float('nan')):.2f}"
                                                   for k in SHOW if k in lg))
        valid = ctl is not None and 0.98 <= ctl <= 1.02
        v["valid"].append(valid)
        if valid:
            v["ii"].append(lg["nvfp4_frozen:rl"]["gain"] >= 0.8)
            v["iii"].append(lg["nvfp4_srfrozen:rl"]["corr"] < lg["nvfp4_frozen:rl"]["corr"])
    if len(v["valid"]) == 2:
        for k in ("ii", "iii"):
            print(f"({k}) {'MET' if len(v[k]) == 2 and all(v[k]) else 'NOT MET'} {v[k]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
