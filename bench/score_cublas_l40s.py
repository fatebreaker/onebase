"""STOCK-CUBLAS-L40S (docs/REGISTRY.md, registered before any data; written before any data): the L40S batch-1 comparison against
vLLM's strongest base, its --linear-backend torch (cuBLASLt) (mlsys jobs/pod_runs/e2e_cublas_l40s_pod.sh -> results/e2e_cublas_l40s
.jsonl; means of the three measured starts per configuration; Qwen3.5-9B on vLLM's online FP8, one L40S, all-module LoRA):
  (i)  with one rank-64 adapter, the plugin's batch-1 time per output token below vLLM's tuned LoRA path on the cuBLASLt base
       (W1(plugin+r64:n1) < W1(stock-tuned-torch+r64:n1))
  (ii) the plugin's one-adapter batch-1 cost over its own base at most 0.6x vLLM's over its own (cuBLASLt):
       W1(plugin+r64:n1)/W1(plugin-noreq+r64:n1) - 1 <= 0.6 x [W1(stock-tuned-torch+r64:n1)/W1(stock-tuned-torch-noreq+r64:n1) - 1]
Descriptive: both paths against the no-LoRA engine on the cuBLASLt base and on vLLM's default base (W1 and W2), and W2 with 32
tenants (16:32, 64:32).
Amendment -MARLIN (written before any data; v21 mock review R1 M1): vLLM's weight-only Marlin FP8 base with the tuned LoRA kernels,
measured with the plugin again (plugin-m) on one GPU:
  (iii) W1(plugin-m+r64:n1) <= W1(stock-tuned-marlin+r64:n1): the plugin at or below vLLM's Marlin path with one adapter at batch 1
Descriptive: the Marlin path's cost over its own base (stock-tuned-marlin-noreq) and the Marlin base without LoRA (nolora-marlin).
    python3 bench/score_cublas_l40s.py [--json results/cublas_l40s_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/e2e_cublas_l40s.jsonl"
REPS = 3
LABELS = ("nolora-default", "nolora-torch", "stock-tuned-torch+r64:n1", "stock-tuned-torch-noreq+r64:n1", "plugin+r64:n1",
          "plugin-noreq+r64:n1", "stock-tuned-torch+r16:n32", "plugin+r16:n32", "stock-tuned-torch+r64:n32", "plugin+r64:n32")
MARLIN = ("nolora-marlin", "stock-tuned-marlin+r64:n1", "stock-tuned-marlin-noreq+r64:n1", "plugin-m+r64:n1")   # -MARLIN


def means(labels: tuple = LABELS) -> dict:
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        lab = r.get("label", "").rsplit("#", 1)[0]
        if lab in labels and "W1_tpot_ms" in r and "W2_gen_tok_s" in r:
            got.setdefault(lab, []).append((r["W1_tpot_ms"], r["W2_gen_tok_s"]))
    return {lab: ((sum(a for a, _ in v[-REPS:]) / REPS, sum(b for _, b in v[-REPS:]) / REPS) if len(v) >= REPS else None)
            for lab, v in ((lab, got.get(lab, [])) for lab in labels)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    m = means()
    # (i) and (ii) need only the 64:1 rows and both no-LoRA bases; 16:32 and 64:32 (unit B) are descriptive (v21: judged without them)
    REQ = ("nolora-default", "nolora-torch", "stock-tuned-torch+r64:n1", "stock-tuned-torch-noreq+r64:n1", "plugin+r64:n1",
           "plugin-noreq+r64:n1")
    if any(m[k] is None for k in REQ):
        res = {"status": "incomplete", "have": [k for k, v in m.items() if v is not None]}
    else:
        w1 = {k: v[0] for k, v in m.items() if v}
        w2 = {k: v[1] for k, v in m.items() if v}
        c_ours = w1["plugin+r64:n1"] / w1["plugin-noreq+r64:n1"] - 1
        c_vllm = w1["stock-tuned-torch+r64:n1"] / w1["stock-tuned-torch-noreq+r64:n1"] - 1
        res = {"W1_ms": w1, "W2_tok_s": w2, "adapter_cost_over_own_base": {"plugin": c_ours, "stock-tuned-torch": c_vllm},
               "vs_nolora_torch_W1": {k: w1[k] / w1["nolora-torch"] - 1 for k in w1 if k != "nolora-torch"},
               "vs_nolora_torch_W2": {k: w2[k] / w2["nolora-torch"] - 1 for k in w2 if k != "nolora-torch"},
               "verdict": {"i": {"plugin_ms": w1["plugin+r64:n1"], "vllm_ms": w1["stock-tuned-torch+r64:n1"],
                                 "met": w1["plugin+r64:n1"] < w1["stock-tuned-torch+r64:n1"]},
                           "ii": {"ratio": c_ours / c_vllm if c_vllm > 0 else None, "met": c_vllm > 0 and c_ours <= 0.6 * c_vllm}}}
    mm = means(MARLIN)                                   # -MARLIN: scored on its own, whatever the main test's state
    if all(v is not None for v in mm.values()):
        w = {k: v[0] for k, v in mm.items()}
        res["marlin"] = {"W1_ms": w, "cost_over_own_base": w["stock-tuned-marlin+r64:n1"] / w["stock-tuned-marlin-noreq+r64:n1"] - 1,
                         "iii": {"plugin_ms": w["plugin-m+r64:n1"], "marlin_ms": w["stock-tuned-marlin+r64:n1"],
                                 "met": w["plugin-m+r64:n1"] <= w["stock-tuned-marlin+r64:n1"]}}
    else:
        res["marlin"] = {"status": "incomplete", "have": [k for k, v in mm.items() if v is not None]}
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
