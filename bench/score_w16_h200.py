"""W16-DECODE-H200 (docs/REGISTRY.md, registered before any data; this copy of bench/score_w16_h100.py written before any data):
the H100 test on an H200 (results/e2e_w16_h200.jsonl), the same two predictions and descriptive rows. The H100 scorer's doc follows.
W16-DECODE-H100 / STOCK-BASES-H100 (docs/REGISTRY.md, registered before any data; written before any data): one adapter
(rank 64) at batch 1 on one H100, Qwen3.5-9B, every base of vLLM's in one job (mlsys jobs/pod_runs/e2e_w16_h100_pod.sh ->
results/e2e_w16_h100.jsonl; means of three measured starts):
  (i)  W1(plugin-w16+r64:n1) < min over vLLM's bases b of W1(stock-tuned-b+r64:n1), b in {default (CUTLASS), torch (cuBLASLt),
       marlin}: the plugin's weight-only decode path below vLLM's fastest stock path on the H100
  (ii) W1(plugin-pdl+r64:n1) < the same minimum: Table 2's H100 plugin (its PDL decode path) below it too
Descriptive: each path's cost over its own base (-noreq); the no-LoRA engine on each base (mock review R1 v22, question 3); W2 at
64:1 and W1/W2 with 32 rank-64 tenants (64:32). If the parity check failed, (i) is void (no plugin-w16 rows).
    python3 bench/score_w16_h200.py [--json results/w16_h200_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_w16_h200.jsonl"
REPS = 3
BASES = {"default": "stock-tuned", "torch": "stock-tuned-torch", "marlin": "stock-tuned-marlin"}
LABELS = ("nolora-default", "nolora-torch", "nolora-marlin",
          *(f"{m}{x}+r64:n1" for m in (*BASES.values(), "plugin-pdl", "plugin-w16") for x in ("", "-noreq")),
          *(f"{m}+r64:n32" for m in (*BASES.values(), "plugin-pdl", "plugin-w16")))


def means() -> dict:
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        lab = r.get("label", "").rsplit("#", 1)[0]
        if lab in LABELS and "W1_tpot_ms" in r and "W2_gen_tok_s" in r:
            got.setdefault(lab, []).append((r["W1_tpot_ms"], r["W2_gen_tok_s"]))
    return {lab: ((sum(a for a, _ in v[-REPS:]) / REPS, sum(b for _, b in v[-REPS:]) / REPS) if len(v) >= REPS else None)
            for lab, v in ((lab, got.get(lab, [])) for lab in LABELS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    m = means()
    stock = [f"{b}+r64:n1" for b in BASES.values()]
    if any(m[k] is None for k in stock) or m["plugin-pdl+r64:n1"] is None:
        res = {"status": "incomplete", "have": [k for k, v in m.items() if v is not None]}
    else:
        w1 = {k: v[0] for k, v in m.items() if v}
        w2 = {k: v[1] for k, v in m.items() if v}
        best = min(stock, key=lambda k: w1[k])
        cost = {k.split("+")[0]: w1[k] / w1[k.replace("+r64:n1", "-noreq+r64:n1")] - 1
                for k in (*stock, "plugin-pdl+r64:n1", "plugin-w16+r64:n1")
                if k in w1 and k.replace("+r64:n1", "-noreq+r64:n1") in w1}
        v = {"fastest_stock": best, "fastest_stock_ms": w1[best],
             "ii": {"plugin_pdl_ms": w1["plugin-pdl+r64:n1"], "ratio": w1["plugin-pdl+r64:n1"] / w1[best],
                    "met": w1["plugin-pdl+r64:n1"] < w1[best]}}
        if "plugin-w16+r64:n1" in w1:
            v["i"] = {"plugin_w16_ms": w1["plugin-w16+r64:n1"], "ratio": w1["plugin-w16+r64:n1"] / w1[best],
                      "met": w1["plugin-w16+r64:n1"] < w1[best]}
        else:
            v["i"] = {"status": "void (no plugin-w16 rows: parity check failed or not run)"}
        res = {"status": "complete", "W1_ms": w1, "W2_tok_s": w2, "cost_over_own_base": cost, "verdict": v}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
