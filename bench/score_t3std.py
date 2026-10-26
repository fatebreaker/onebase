"""T3-STD (docs/REGISTRY.md, registered 2026-10-14 03:23 UTC): the output gain gamma and weight gain g_W of each standard
adapter's nearest frozen NVFP4 merge (mlsys results/census/t3std_<model>_<which>.jsonl, the nvfp4_frozen summary row)
against its measured logit gain g (results/std_bench/srmerge_<model>_<which>.json, 'nvfp4_frozen:rl').
  (i)   valid if the Llama copy control's g is in [0.98, 1.02]
  (ii)  |gamma - g| <= 0.15 on at least 2 of the 3 adapters, and median |gamma - g| < median |g_W - g|
  (iii) the Llama math adapter keeps g <= 0.3; gamma <= 0.3 for it and >= 0.7 for both Mistral adapters
    python bench/score_t3std.py [--json out.json]"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
ML = Path("$ROUNDING")
ADAPTERS = (("mistral", "math"), ("mistral", "code"), ("llama", "math"))


def census(m: str, w: str) -> dict | None:
    f = ML / "results/census" / f"t3std_{m}_{w}.jsonl"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r.get("summary") and r.get("scheme") == "nvfp4_frozen":
            return r
    return None


def logits(m: str, w: str) -> tuple[float | None, float | None]:
    f = LB / "results/std_bench" / f"srmerge_{m}_{w}.json"
    if not f.exists():
        return None, None
    lg = json.loads(f.read_text())["logit_gain"]
    ctl = next((x["gain"] for k, x in lg.items() if k.startswith("lora=") and k.endswith("/copy")), None)
    return lg["nvfp4_frozen:rl"]["gain"], ctl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = {}
    for m, w in ADAPTERS:
        c = census(m, w)
        g, ctl = logits(m, w)
        if c is None or g is None:
            print(f"{m} {w}: incomplete (census {c is not None}, logits {g is not None})")
            continue
        rows[f"{m}_{w}"] = {"g": g, "control": ctl, "g_W": c["gain"], "gamma": c["out_gain"]}
        print(f"{m:8s} {w:5s} logit g {g:.3f} (control {ctl:.3f})  output gain {c['out_gain']:.3f}  weight gain {c['gain']:.3f}")
    if len(rows) < 3:
        return 1
    ctl = rows["llama_math"]["control"]
    valid = ctl is not None and 0.98 <= ctl <= 1.02
    eg = [abs(r["gamma"] - r["g"]) for r in rows.values()]
    ew = [abs(r["g_W"] - r["g"]) for r in rows.values()]
    ii = sum(e <= 0.15 for e in eg) >= 2 and statistics.median(eg) < statistics.median(ew)
    iii = (rows["llama_math"]["g"] <= 0.3 and rows["llama_math"]["gamma"] <= 0.3
           and all(rows[k]["gamma"] >= 0.7 for k in ("mistral_math", "mistral_code")))
    v = {"valid": valid, "ii": ii, "iii": iii, "abs_err_gamma": eg, "abs_err_gW": ew}
    print(f"(i) {'valid' if valid else 'INVALID'}; (ii) {'MET' if ii else 'NOT MET'} (|gamma-g| {[round(e, 3) for e in eg]}, "
          f"|g_W-g| {[round(e, 3) for e in ew]}); (iii) {'MET' if iii else 'NOT MET'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
