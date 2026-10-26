"""SERVE-OWNBASE (docs/REGISTRY.md, registered before any data; v14 mock review R1 M1): where the H100's and the Blackwell slice's
batch-1 gains come from. Per GPU, one job measures, with three measured starts each (means over them): the no-LoRA engine; vLLM's
LoRA path with its tuned kernels (stock-tuned) and the plugin with its decode path (plugin-pdl), each with one rank-64 adapter
(64:1) and each with the same adapter loaded but every request without one (*-noreq: the path's own base inside LoRA layers).
Rows: results/e2e_ownbase_bw.jsonl (jobs/e2e_ownbase_bw.sbatch) and results/e2e_ownbase_h100.jsonl (mlsys
jobs/pod_runs/e2e_ownbase_h100_pod.sh); labels <mode>+r64:n1#<start>, nolora#<start>.
  (i) per GPU: the plugin's one-adapter batch-1 cost over its own base, W1(plugin-pdl) / W1(plugin-pdl-noreq) - 1, at most 0.6x
      vLLM's over its own, W1(stock-tuned) / W1(stock-tuned-noreq) - 1
Descriptive: each path's own base against the no-LoRA engine (W1 and W2), and each path with the adapter against the no-LoRA engine.
A GPU is scored once all five configurations have three measured starts.
    python3 bench/score_ownbase.py [--json results/ownbase_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
FILES = {"blackwell": LB / "results/e2e_ownbase_bw.jsonl", "h100": LB / "results/e2e_ownbase_h100.jsonl"}
LABELS = ("nolora", "stock-tuned+r64:n1", "stock-tuned-noreq+r64:n1", "plugin-pdl+r64:n1", "plugin-pdl-noreq+r64:n1")
REPS = 3


def means(path: Path) -> dict:
    """{label: (W1 ms, W2 tokens/s)} over the measured starts (rows with both workloads); None if fewer than REPS."""
    got: dict = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        lab = r.get("label", "").rsplit("#", 1)[0]
        if lab in LABELS and "W1_tpot_ms" in r and "W2_gen_tok_s" in r:
            got.setdefault(lab, []).append((r["W1_tpot_ms"], r["W2_gen_tok_s"]))
    return {k: (sum(a for a, _ in v[-REPS:]) / REPS, sum(b for _, b in v[-REPS:]) / REPS) if len(v) >= REPS else None
            for k, v in ((lab, got.get(lab, [])) for lab in LABELS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res: dict = {}
    for gpu, f in FILES.items():
        m = means(f)
        if any(v is None for v in m.values()):
            res[gpu] = {"status": "incomplete", "have": [k for k, v in m.items() if v is not None]}
            print(gpu, "incomplete:", res[gpu]["have"])
            continue
        w1 = {k: v[0] for k, v in m.items()}
        w2 = {k: v[1] for k, v in m.items()}
        c_ours = w1["plugin-pdl+r64:n1"] / w1["plugin-pdl-noreq+r64:n1"] - 1
        c_vllm = w1["stock-tuned+r64:n1"] / w1["stock-tuned-noreq+r64:n1"] - 1
        d = {"W1_ms": w1, "W2_tok_s": w2,
             "adapter_cost_over_own_base": {"plugin-pdl": c_ours, "stock-tuned": c_vllm},
             "own_base_vs_nolora_W1": {"plugin-pdl": w1["plugin-pdl-noreq+r64:n1"] / w1["nolora"] - 1,
                                       "stock-tuned": w1["stock-tuned-noreq+r64:n1"] / w1["nolora"] - 1},
             "own_base_vs_nolora_W2": {"plugin-pdl": w2["plugin-pdl-noreq+r64:n1"] / w2["nolora"] - 1,
                                       "stock-tuned": w2["stock-tuned-noreq+r64:n1"] / w2["nolora"] - 1},
             "with_adapter_vs_nolora_W1": {"plugin-pdl": w1["plugin-pdl+r64:n1"] / w1["nolora"] - 1,
                                           "stock-tuned": w1["stock-tuned+r64:n1"] / w1["nolora"] - 1},
             "verdict": {"i": {"ratio": c_ours / c_vllm if c_vllm > 0 else None,
                               "met": c_vllm > 0 and c_ours <= 0.6 * c_vllm}}}
        res[gpu] = d
        print(gpu, json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) and k != "verdict" else v)
                               for k, v in d.items()}, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
