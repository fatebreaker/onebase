"""THRU-ATTR-H100 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data): where the plugin's 32-tenant
throughput gain on the H100 comes from. Rows: results/thru_attr_h100.jsonl (mlsys jobs/pod_runs/thru_attr_h100_pod.sh), means of the
three measured starts (labels '#1'-'#3'); kernels: results/pdl_k2_attr32_h100.jsonl (stage attr32, the newest job).
  (i)  W2(plugin-hybrid+r64:n32) >= 1.05 x max(W2(stock-tuned+r64:n32), W2(stock-dual-tuned+r64:n32))
  (ii) |W2(plugin-pdl+r64:n32) / W2(plugin-hybrid+r64:n32) - 1| <= 0.03
Descriptive: every mode's W2 and W1 at 64:32 and 16:32 against stock-tuned and nolora; per projection and M, each path's LoRA cost
(t_<path> - t_ref) and the hybrid path's share of vLLM's tuned one, summed over a Qwen3.5-9B layer of each kind.
    python3 bench/score_thru_attr.py [--json results/thru_attr_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/thru_attr_h100.jsonl"
K = LB / "results/pdl_k2_attr32_h100.jsonl"
MODES = ("stock", "stock-tuned", "stock-dual-tuned", "plugin-hybrid", "plugin-auto", "plugin-pdl")
PATHS = ("stock", "stock_tuned", "stock_dual", "hybrid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    got: dict = {}
    for line in F.read_text().splitlines() if F.exists() else []:
        if line.strip():
            r = json.loads(line)
            lab, _, rep = r["label"].rpartition("#")
            if rep in ("1", "2", "3"):
                got.setdefault(lab, {})[rep] = r
    m = {k: {w: sum(x[w] for x in v.values()) / len(v) for w in ("W1_tpot_ms", "W2_gen_tok_s")} for k, v in got.items() if len(v) == 3}
    out: dict = {"means": m}
    need = [f"{x}+r64:n32" for x in ("stock-tuned", "stock-dual-tuned", "plugin-hybrid", "plugin-pdl")]
    out["complete"] = all(k in m for k in need)
    if out["complete"]:
        best = max(m["stock-tuned+r64:n32"]["W2_gen_tok_s"], m["stock-dual-tuned+r64:n32"]["W2_gen_tok_s"])
        hyb = m["plugin-hybrid+r64:n32"]["W2_gen_tok_s"]
        pdl = m["plugin-pdl+r64:n32"]["W2_gen_tok_s"]
        out["i"] = {"met": hyb >= 1.05 * best, "ratio": hyb / best, "hybrid": hyb, "vllm_best": best}
        out["ii"] = {"met": abs(pdl / hyb - 1) <= 0.03, "ratio": pdl / hyb}
    for s in ("64:32", "16:32"):
        ref = m.get(f"stock-tuned+r{s.split(':')[0]}:n32")
        if ref:
            out[f"vs_stock_tuned_{s}"] = {md: {"W2": m[f"{md}+r{s.split(':')[0]}:n32"]["W2_gen_tok_s"] / ref["W2_gen_tok_s"],
                                               "W1": m[f"{md}+r{s.split(':')[0]}:n32"]["W1_tpot_ms"] / ref["W1_tpot_ms"]}
                                          for md in MODES if f"{md}+r{s.split(':')[0]}:n32" in m}
    if "nolora-default" in m:
        out["nolora"] = m["nolora-default"]
    rows = [json.loads(x) for x in K.read_text().splitlines() if x.strip()] if K.exists() else []
    rows = [r for r in rows if r.get("stage") == "attr32"]
    if rows:
        job = max(str(r.get("job", "")) for r in rows)
        rows = [r for r in rows if str(r.get("job", "")) == job]
        kern: dict = {}
        for r in rows:
            if not isinstance(r.get("t_ref"), (int, float)):
                continue
            kern.setdefault(str(r["M"]), {})[r["proj"]] = {p: r.get(f"t_{p}") - r["t_ref"] for p in PATHS
                                                           if isinstance(r.get(f"t_{p}"), (int, float))} | {"ref": r["t_ref"]}
        out["kernel_lora_cost_us"] = kern
        lay = {}
        for M, d in kern.items():
            for kind, projs in (("full_attn", ("qkv", "o", "gate_up", "down")), ("linear_attn", ("qkvz", "o", "gate_up", "down"))):
                if all(p in d for p in projs) and all(all(q in d[p] for q in ("stock_tuned", "hybrid")) for p in projs):
                    st = sum(d[p]["stock_tuned"] for p in projs)
                    hy = sum(d[p]["hybrid"] for p in projs)
                    lay[f"M{M}_{kind}"] = {"stock_tuned": st, "hybrid": hy, "hybrid_over_stock_tuned": hy / st if st else None}
        out["layer_lora_cost_us"] = lay
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
