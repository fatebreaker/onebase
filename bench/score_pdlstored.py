"""HOPPER-PDL-STORED (docs/REGISTRY.md; this scorer written before any data): the plugin's PDL decode path with FP8- and
NVFP4-stored adapters end to end on one H100, from results/e2e_pdlstored.jsonl (mlsys jobs/pod_runs/e2e_pdlst_pod.sh; labels
<mode>+r<r>:n<n>#<start> and nolora#<start>; per configuration the mean of its last three measured starts within one run, as
bench/score_stocktuned.load). Modes: plugin-pdl (bf16 adapters, this job's reference), plugin-pdl-fp8, plugin-pdl-fp4; settings
with LoRA (rank, adapters): (64, 1), (16, 32), (64, 32).
  (i)   every configuration has three measured starts (else incomplete)
  (ii)  plugin-pdl-fp4's W1 (batch-1 time per output token) at 64:1 at most 1.15x plugin-pdl's
  (iii) plugin-pdl-fp8's W1 at most 1.05x plugin-pdl's at every setting with LoRA
Descriptive: W1 and W2 of every mode and setting and their change against the no-LoRA engine; each stored mode against
HOPPER-PDL-E2E's vLLM configurations (results/std_bench/hopperpdl_verdict.json: the better of stock-tuned and stock-dual, a
separate job) and against the plugin's NVFP4-stored adapters without the decode path (E2E-QWEN-H100,
results/e2e_warm_qwen9_h100.jsonl, one start); the cold starts' output comparison with HOPPER-PDL-E2E's plugin-pdl dumps.
    python3 bench/score_pdlstored.py [--json results/std_bench/pdlstored_verdict.json]
BLACKWELL-PDL-STORED applies the same predictions on the Blackwell slice (our NVFP4 Qwen3-8B; BLACKWELL-PDL-E2E's vLLM
configurations; the NVFP4-stored plugin without the decode path from blackwell_warm_qwen3.jsonl):
    python3 bench/score_pdlstored.py --platform bw [--json results/std_bench/bwpdlstored_verdict.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_stocktuned import load, mean  # noqa: E402

MODES = ("plugin-pdl", "plugin-pdl-fp8", "plugin-pdl-fp4")
SETTINGS = ((64, 1), (16, 32), (64, 32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--platform", choices=("h100", "bw"), default="h100")
    ap.add_argument("--data", default="")
    a = ap.parse_args()
    a.data = a.data or {"h100": "results/e2e_pdlstored.jsonl", "bw": "results/e2e_bwpdlstored.jsonl"}[a.platform]
    ref_verdict = {"h100": "hopperpdl_verdict.json", "bw": "bwpdl_verdict.json"}[a.platform]
    grid = {"h100": ("NVIDIA H100 80GB HBM3", "Qwen3.5-9B-text", "fp8", "e2e_warm_qwen9_h100.jsonl"),
            "bw": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "Qwen3-8B", "modelopt_fp4", "blackwell_warm_qwen3.jsonl")}[a.platform]
    need = [(m, s) for m in MODES for s in SETTINGS] + [("nolora", None)]
    d = load(LB / a.data, need) if (LB / a.data).exists() else {}
    miss = [k for k in need if len(d.get(k, {}).get("W1", [])) < 3]
    out: dict = {"data": a.data}
    e = {k: {w: mean(v[w]) for w in ("W1", "W2")} for k, v in d.items() if len(v.get("W1", [])) >= 3}
    ref = e.get(("nolora", None))
    hp = LB / "results/std_bench" / ref_verdict
    hpv = json.loads(hp.read_text()) if hp.exists() else {}
    per: dict = {}
    for s in SETTINGS:
        lab = f"{s[0]}:{s[1]}"
        p = {f"{m}_{w}": e[(m, s)][w] for m in MODES for w in ("W1", "W2") if (m, s) in e}
        if ref:
            p.update({f"{m}_{w}_change_pct": 100 * (e[(m, s)][w] / ref[w] - 1)
                      for m in MODES for w in ("W1", "W2") if (m, s) in e})
        for m in MODES[1:]:
            if (m, s) in e and ("plugin-pdl", s) in e:
                p[f"{m}_over_bf16_W1"] = e[(m, s)]["W1"] / e[("plugin-pdl", s)]["W1"]
                p[f"{m}_over_bf16_W2"] = e[(m, s)]["W2"] / e[("plugin-pdl", s)]["W2"]
        hs = (hpv.get("settings") or {}).get(lab, {})
        hn = hpv.get("nolora") or (hpv.get("verdict") or {}).get("nolora")
        if hs and hn:                                  # HOPPER-PDL-E2E's vLLM, as changes against its own no-LoRA engine
            p["vllm_best_W1_change_pct_e2epdl"] = min(100 * (hs[f"{m}_W1"] / hn["W1"] - 1) for m in ("stock-tuned", "stock-dual"))
            p["vllm_best_W2_change_pct_e2epdl"] = max(100 * (hs[f"{m}_W2"] / hn["W2"] - 1) for m in ("stock-tuned", "stock-dual"))
        per[lab] = p
    out["per_setting"] = per
    out["nolora"] = ref
    try:                                               # the NVFP4-stored plugin without the decode path (one start)
        sys.path.insert(0, str(LB / "paper/figs"))
        import make_figs as mf
        g = mf.e2e_platform(grid[0], grid[1], grid[2], files=(grid[3],))
        out["fp4_without_pdl_change_pct"] = {f"{r}:{n}": {w: g[("fused-fp4ad", (r, n))][w] for w in ("W1", "W2")}
                                             for r, n in SETTINGS if ("fused-fp4ad", (r, n)) in g}
    except Exception as ex:   # noqa: BLE001 -- descriptive only
        out["fp4_without_pdl_change_pct"] = f"unavailable: {ex}"
    v: dict = {}
    v["i"] = "met" if not miss else "incomplete"
    if miss:
        out["missing"] = [f"{m}+r{s[0]}:n{s[1]}" if s else m for m, s in miss]
    r2 = per.get("64:1", {}).get("plugin-pdl-fp4_over_bf16_W1")
    v["ii"] = "incomplete" if r2 is None else ("met" if r2 <= 1.15 else "not met")
    r3 = {lab: per[lab].get("plugin-pdl-fp8_over_bf16_W1") for lab in per}
    if any(x is None for x in r3.values()):
        v["iii"] = "incomplete"
    else:
        bad = [lab for lab, x in r3.items() if x > 1.05]
        v["iii"] = "met" if not bad else f"not met at {', '.join(bad)}"
    out["verdict"] = v
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
