"""STOCK-TUNED-TM (docs/REGISTRY.md, registered 2026-10-19 00:44 UTC, before any data): the strongest stock path, vLLM's LoRA
kernels tuned for the GPU with LoRA only on the adapters' projections (--lora-target-modules), against the plugin with the same
target modules (jobs/e2e_stocktuned_tm.sbatch -> results/e2e_stocktuned_tm.jsonl, labels <mode>+r<r>:n<n>#<start> and
nolora#<start>; the last three starts of each configuration, as bench/score_stocktuned.py):
  (i)   the plugin's (auto-tm) mean W1 (batch-1 time per output token) below stock-tuned-tm's at (64, 1), (16, 32) and (64, 32)
  (ii)  the plugin's mean W2 (throughput) above stock-tuned-tm's at the same settings
Descriptive: fp4-tm (NVFP4-stored adapters) against stock-tuned-tm at 32 tenants; every mode against the no-LoRA engine.
    python3 bench/score_stocktuned_tm.py [--json results/stocktuned_tm_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_stocktuned import SETTINGS, load, mean  # noqa: E402

MODES = ("stock-tm", "stock-tuned-tm", "auto-tm", "fp4-tm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    need = [(m, s) for m in MODES[:3] for s in SETTINGS] + [("nolora", None)]
    d = load(LB / "results/e2e_stocktuned_tm.jsonl", need)
    miss = [k for k in need if len(d.get(k, {}).get("W1", [])) < 3]
    if miss:
        print(f"incomplete: {len(miss)} configurations missing, e.g. {miss[0]}")
        if a.json:
            Path(a.json).write_text(json.dumps({"status": "incomplete", "missing": [str(k) for k in miss]}, indent=1))
        return 1
    ref = {w: mean(d[("nolora", None)][w]) for w in ("W1", "W2")}
    per = {}
    for s in SETTINGS:
        e = {m: {w: mean(d[(m, s)][w]) for w in ("W1", "W2")} for m in MODES if len(d.get((m, s), {}).get("W1", [])) >= 3}
        p = {**{f"{m}_{w}": e[m][w] for m in e for w in ("W1", "W2")},
             **{f"{m}_{w}_change": 100 * (e[m][w] / ref[w] - 1) for m in e for w in ("W1", "W2")},
             "auto_vs_tuned_W1": e["auto-tm"]["W1"] / e["stock-tuned-tm"]["W1"],
             "auto_vs_tuned_W2": e["auto-tm"]["W2"] / e["stock-tuned-tm"]["W2"]}
        if "fp4-tm" in e:
            p["fp4_vs_tuned_W1"] = e["fp4-tm"]["W1"] / e["stock-tuned-tm"]["W1"]
            p["fp4_vs_tuned_W2"] = e["fp4-tm"]["W2"] / e["stock-tuned-tm"]["W2"]
        per[f"{s[0]}:{s[1]}"] = p
    v = {"i": {"met": all(p["auto-tm_W1"] < p["stock-tuned-tm_W1"] for p in per.values())},
         "ii": {"met": all(p["auto-tm_W2"] > p["stock-tuned-tm_W2"] for p in per.values())},
         "job": d[("nolora", None)]["job"], "nolora": ref}
    for s, p in per.items():
        print(s, json.dumps({k: round(x, 3) for k, x in p.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"settings": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
