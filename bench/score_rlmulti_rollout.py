"""RL-MULTI-ROLLOUT (docs/REGISTRY.md, registered before any data; written before any data): rollout throughput for K users' LoRA RL
policies on one FP8 engine (mlsys jobs/pod_runs/rlmulti_rollout_pod.sh -> results/rl_multi_rollout_<gpu>.jsonl; the newest row per
label; output tokens/s, best of three repeats of fixed work). Per GPU (L40S, H100):
  (i)  with K = 32 policies, the plugin's rollout throughput at least 1.05x that of vLLM's LoRA path with tuned kernels
Descriptive: K = 1 and 8, each variant against the engine without LoRA.
    python3 bench/score_rlmulti_rollout.py [--json results/rl_multi_rollout_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]


def rows(gpu: str) -> dict:
    f = LB / f"results/rl_multi_rollout_{gpu}.jsonl"
    out = {}
    for line in f.read_text().splitlines() if f.exists() else []:
        if line.strip():
            r = json.loads(line)
            out[r["label"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    res = {}
    for gpu in ("l40s", "h100"):
        r = rows(gpu)
        # the prediction needs only K = 32 (both variants); K = 1, 8 and the no-LoRA engine are descriptive (v21: the L40S's
        # descriptive unit D waited for a free GPU, so (i) is judged from K = 32 alone)
        if any(f"{v}+K32" not in r for v in ("stock-tuned", "plugin")):
            res[gpu] = {"status": "incomplete", "have": sorted(r)}
            continue
        t = {n: x["tok_per_s"] for n, x in r.items()}
        ks = [k for k in (1, 8, 32) if f"plugin+K{k}" in t and f"stock-tuned+K{k}" in t]
        ratio = {k: t[f"plugin+K{k}"] / t[f"stock-tuned+K{k}"] for k in ks}
        res[gpu] = {"tok_s": t, "plugin_over_stock": ratio,
                    "vs_nolora": {n: t[n] / t["nolora+K0"] for n in t if n != "nolora+K0"} if "nolora+K0" in t else {},
                    "verdict": {"i": {"ratio": ratio[32], "met": ratio[32] >= 1.05}}}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
