"""STOCK-TUNED-TM-H100 (docs/REGISTRY.md, registered before any data): the strongest stock configuration on the H100, vLLM's
H100-tuned LoRA kernels with LoRA only on the adapters' projections (--lora-target-modules), against the plugin with its PDL decode
path under the same flag (mlsys jobs/pod_runs/e2e_tm_h100_pod.sh -> results/e2e_tm_h100.jsonl, labels <mode>+r<r>:n<n>#<start> and
nolora#<start>; the last three starts of each configuration, as bench/score_stocktuned.py):
  (i)   plugin-pdl-tm's mean W1 (batch-1 time per output token) below stock-tuned-tm's at (64, 1), (16, 32) and (64, 32)
  (ii)  plugin-pdl-tm's mean W2 (throughput) at least 0.98x stock-tuned-tm's at the same settings
Descriptive: both modes against the no-LoRA engine.
    python3 bench/score_tm_h100.py [--json results/tm_h100_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_stocktuned import SETTINGS, load, mean  # noqa: E402

MODES = ("stock-tuned-tm", "plugin-pdl-tm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    need = [(m, s) for m in MODES for s in SETTINGS] + [("nolora", None)]
    d = load(LB / "results/e2e_tm_h100.jsonl", need)
    miss = [k for k in need if len(d.get(k, {}).get("W1", [])) < 3]
    if miss:
        print(f"incomplete: {len(miss)} configurations missing, e.g. {miss[0]}")
        if a.json:
            Path(a.json).write_text(json.dumps({"status": "incomplete", "missing": [str(k) for k in miss]}, indent=1))
        return 1
    ref = {w: mean(d[("nolora", None)][w]) for w in ("W1", "W2")}
    per = {}
    for s in SETTINGS:
        e = {m: {w: mean(d[(m, s)][w]) for w in ("W1", "W2")} for m in MODES}
        per[f"{s[0]}:{s[1]}"] = {**{f"{m}_{w}": e[m][w] for m in e for w in ("W1", "W2")},
                                 **{f"{m}_{w}_change": 100 * (e[m][w] / ref[w] - 1) for m in e for w in ("W1", "W2")},
                                 "pdl_vs_tuned_W1": e["plugin-pdl-tm"]["W1"] / e["stock-tuned-tm"]["W1"],
                                 "pdl_vs_tuned_W2": e["plugin-pdl-tm"]["W2"] / e["stock-tuned-tm"]["W2"]}
    v = {"i": {"met": all(p["plugin-pdl-tm_W1"] < p["stock-tuned-tm_W1"] for p in per.values())},
         "ii": {"met": all(p["pdl_vs_tuned_W2"] >= 0.98 for p in per.values())},
         "job": d[("nolora", None)]["job"], "nolora": ref}
    for s, p in per.items():
        print(s, json.dumps({k: round(x, 3) for k, x in p.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"settings": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
