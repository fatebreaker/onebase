"""STOCK-TUNED-H100-AT (docs/REGISTRY.md, registered before any data): the plugin with vLLM's H100-tuned LoRA configurations as its
dispatch's stock candidate (auto-tuned) against vLLM's tuned path (stock-tuned), from results/e2e_stocktuned_h100_at.jsonl (labels
<mode>+r<r>:n<n>#<start>, nolora#<start>; each configuration's last three measured starts, as bench/score_stocktuned.py):
  (i)   auto-tuned's mean W1 (batch-1 time per output token) at most 1.02x stock-tuned's in every setting
  (ii)  auto-tuned's mean W2 (throughput) at least 0.98x stock-tuned's in every setting
Descriptive: auto against stock-tuned and auto-tuned against auto, each mode against the no-LoRA engine.
    python3 bench/score_stocktuned_at.py [--json results/stocktuned_h100_at_verdict.json]
STOCK-TUNED-L40S-AT (registered 2026-10-21 05:19 UTC, the same predictions on the L40S):
    python3 bench/score_stocktuned_at.py --data results/e2e_stocktuned_l40s_at.jsonl --json results/stocktuned_l40s_at_verdict.json"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_stocktuned import SETTINGS, load, mean  # noqa: E402

MODES = ("stock-tuned", "auto", "auto-tuned")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--data", default="results/e2e_stocktuned_h100_at.jsonl")
    a = ap.parse_args()
    need = [(m, s) for m in MODES for s in SETTINGS] + [("nolora", None)]
    d = load(LB / a.data, need)
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
        per[f"{s[0]}:{s[1]}"] = {**{f"{m}_{w}": e[m][w] for m in MODES for w in ("W1", "W2")},
                                 **{f"{m}_{w}_change": 100 * (e[m][w] / ref[w] - 1) for m in MODES for w in ("W1", "W2")},
                                 "at_vs_tuned_W1": e["auto-tuned"]["W1"] / e["stock-tuned"]["W1"],
                                 "at_vs_tuned_W2": e["auto-tuned"]["W2"] / e["stock-tuned"]["W2"],
                                 "auto_vs_tuned_W1": e["auto"]["W1"] / e["stock-tuned"]["W1"],
                                 "auto_vs_tuned_W2": e["auto"]["W2"] / e["stock-tuned"]["W2"]}
    v = {"i": {"met": all(p["at_vs_tuned_W1"] <= 1.02 for p in per.values())},
         "ii": {"met": all(p["at_vs_tuned_W2"] >= 0.98 for p in per.values())},
         "job": d[("nolora", None)]["job"], "nolora": ref}
    for s, p in per.items():
        print(s, json.dumps({k: round(x, 3) for k, x in p.items()}))
    print("verdict:", json.dumps(v))
    if a.json:
        Path(a.json).write_text(json.dumps({"settings": per, "verdict": v}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
