"""W16-DECODE-L40S extension -TM (docs/REGISTRY.md, registered 2026-10-25 before any data; this copy of score_w16_l40s.py written
before any data): the same comparison with vLLM's --lora-target-modules (the adapters' seven projections) in every LoRA mode, its
strongest setting (mock review R1 v26, W3); mlsys jobs/pod_runs/e2e_w16tm_l40s_pod.sh -> results/e2e_w16tm_l40s.jsonl. Only (i) is
registered for -TM; (ii) is descriptive. The original's description follows.
W16-DECODE-L40S (docs/REGISTRY.md, registered before any data; written before any data): the plugin's weight-only decode path
(LOWBIT_A16_MAX_M=32) at batch 1 against vLLM's weight-only Marlin FP8 base with its tuned LoRA kernels, Qwen3.5-9B on one L40S,
all in one job (mlsys jobs/pod_runs/e2e_w16_l40s_pod.sh -> results/e2e_w16_l40s.jsonl; means of three measured starts):
  (i)  W1(plugin-w16+r64:n1) < W1(stock-tuned-marlin+r64:n1): with one rank-64 adapter, the plugin's batch-1 time per output token
       below vLLM's fastest stock path on the L40S
  (ii) its cost over its own base at most 0.6x the Marlin path's over its own:
       W1(plugin-w16+r64:n1)/W1(plugin-w16-noreq+r64:n1) - 1 <= 0.6 x [W1(stock-tuned-marlin+r64:n1)/W1(stock-tuned-marlin-noreq+r64:n1) - 1]
Descriptive: plugin-w16 against plugin-m (the W8A8 plugin, the same job) and the no-LoRA Marlin engine; W1 and W2 with 32 rank-64
tenants (64:32).
    python3 bench/score_w16tm_l40s.py [--json results/w16tm_l40s_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_w16tm_l40s.jsonl"
REPS = 3
LABELS = ("nolora-marlin", "stock-tuned-marlin+r64:n1", "stock-tuned-marlin-noreq+r64:n1", "plugin-w16+r64:n1",
          "plugin-w16-noreq+r64:n1", "stock-tuned-marlin+r64:n32", "plugin-w16+r64:n32")


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
    req = ("stock-tuned-marlin+r64:n1", "stock-tuned-marlin-noreq+r64:n1", "plugin-w16+r64:n1", "plugin-w16-noreq+r64:n1")
    if any(m[k] is None for k in req):
        res = {"status": "incomplete", "have": [k for k, v in m.items() if v is not None]}
    else:
        w1 = {k: v[0] for k, v in m.items() if v}
        w2 = {k: v[1] for k, v in m.items() if v}
        c_ours = w1["plugin-w16+r64:n1"] / w1["plugin-w16-noreq+r64:n1"] - 1
        c_marlin = w1["stock-tuned-marlin+r64:n1"] / w1["stock-tuned-marlin-noreq+r64:n1"] - 1
        res = {"W1_ms": w1, "W2_tok_s": w2, "cost_over_own_base": {"plugin-w16": c_ours, "stock-tuned-marlin": c_marlin},
               "verdict": {"i": {"plugin_w16_ms": w1["plugin-w16+r64:n1"], "marlin_ms": w1["stock-tuned-marlin+r64:n1"],
                                 "ratio": w1["plugin-w16+r64:n1"] / w1["stock-tuned-marlin+r64:n1"],
                                 "met": w1["plugin-w16+r64:n1"] < w1["stock-tuned-marlin+r64:n1"]},
                           "ii": {"ratio": c_ours / c_marlin if c_marlin > 0 else None, "met": c_marlin > 0 and c_ours <= 0.6 * c_marlin}}}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
