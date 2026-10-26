"""HOPPER-PDL-E2E (docs/REGISTRY.md, registered 2026-10-21 21:29 UTC before any data; this scorer written before any data):
the plugin with the opt-in PDL decode path (plugin-pdl) end to end on one H100 against vLLM's LoRA path with its default,
H100-tuned and dual-stream kernels and the plugin without the PDL path, from results/e2e_hopperpdl.jsonl (mlsys
jobs/pod_runs/e2e_pdl_pod.sh; labels <mode>+r<r>:n<n>#<start> and nolora#<start>; per configuration the mean of its last
three measured starts within one run: the newest run holding three starts of every configuration, as
bench/score_stocktuned.load). Settings with LoRA (rank, adapters): (64, 1), (16, 32), (64, 32).
  (i)   plugin-pdl's W1 (batch-1 time per output token) below stock-tuned's and below stock-dual's at every setting with LoRA
  (ii)  plugin-pdl's W1 at 64:1 at most 1.10x the no-LoRA engine's (the same job's)
  (iii) plugin-pdl's W2 (throughput) at least 0.95x the better (higher) of stock-tuned's and stock-dual's at every setting
        with LoRA
Each met at every setting or not met (the settings where it fails listed); incomplete while a configuration lacks three
measured starts. Descriptive: W1 and W2 of every mode and setting and their change against the no-LoRA engine; the output
checks of the cold starts (results/e2e_hopperpdl_cold.jsonl: each mode's 16 prompts against stock's of the same setting,
tokens identical or log-probs within 2e-2; also on strong adapters and at batch 1); the dispatch's choices per layer and
bucket (the plugin modes' cold starts); cold-start (engine load) times.
    python3 bench/score_hopperpdl.py [--json results/std_bench/hopperpdl_verdict.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_stocktuned import load, mean  # noqa: E402

MODES = ("stock", "stock-tuned", "stock-dual", "plugin", "plugin-pdl")
SETTINGS = ((64, 1), (16, 32), (64, 32))


def cold_rows(path: Path, job: str) -> dict:
    """The cold starts of the job: label (without #cold) -> its newest row."""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("job", "") != job or not r.get("label", "").endswith("#cold"):
            continue
        lab = r["label"].rsplit("#", 1)[0]
        if lab not in out or r.get("time", "") >= out[lab].get("time", ""):
            out[lab] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--data", default="results/e2e_hopperpdl.jsonl")
    ap.add_argument("--cold", default="results/e2e_hopperpdl_cold.jsonl")
    a = ap.parse_args()
    need = [(m, s) for m in MODES for s in SETTINGS] + [("nolora", None)]
    d = load(LB / a.data, need)
    miss = [k for k in need if len(d.get(k, {}).get("W1", [])) < 3]
    out: dict = {"data": a.data}
    if miss:
        out["status"] = "incomplete"
        out["missing"] = [f"{m}+r{s[0]}:n{s[1]}" if s else m for m, s in miss]
        print(f"incomplete: {len(miss)} configurations without three measured starts, e.g. {out['missing'][0]}")
    job = next(iter(d.values()))["job"] if d else ""
    e = {k: {w: mean(v[w]) for w in ("W1", "W2")} for k, v in d.items() if len(v.get("W1", [])) >= 3}
    ref = e.get(("nolora", None))
    per: dict = {}
    for s in SETTINGS:
        lab = f"{s[0]}:{s[1]}"
        p = {f"{m}_{w}": e[(m, s)][w] for m in MODES for w in ("W1", "W2") if (m, s) in e}
        if ref:
            p.update({f"{m}_{w}_change_pct": 100 * (e[(m, s)][w] / ref[w] - 1)
                      for m in MODES for w in ("W1", "W2") if (m, s) in e})
        if all((m, s) in e for m in ("plugin-pdl", "stock-tuned", "stock-dual")):
            pdl, tun, dua = e[("plugin-pdl", s)], e[("stock-tuned", s)], e[("stock-dual", s)]
            p["pdl_over_tuned_W1"] = pdl["W1"] / tun["W1"]
            p["pdl_over_dual_W1"] = pdl["W1"] / dua["W1"]
            p["pdl_W2_over_best_stock_W2"] = pdl["W2"] / max(tun["W2"], dua["W2"])
        if ("plugin", s) in e and ("plugin-pdl", s) in e:
            p["pdl_over_plugin_W1"] = e[("plugin-pdl", s)]["W1"] / e[("plugin", s)]["W1"]
            p["pdl_over_plugin_W2"] = e[("plugin-pdl", s)]["W2"] / e[("plugin", s)]["W2"]
        per[lab] = p
    v: dict = {}
    if "status" not in out:
        f1 = [lab for lab, p in per.items() if not (p["pdl_over_tuned_W1"] < 1 and p["pdl_over_dual_W1"] < 1)]
        v["i"] = {"met": not f1, "fails": f1,
                  "pdl_over_tuned_W1": {lab: p["pdl_over_tuned_W1"] for lab, p in per.items()},
                  "pdl_over_dual_W1": {lab: p["pdl_over_dual_W1"] for lab, p in per.items()}}
        r2 = e[("plugin-pdl", (64, 1))]["W1"] / ref["W1"]
        v["ii"] = {"met": r2 <= 1.10, "pdl_W1_64_1_over_nolora": r2}
        f3 = [lab for lab, p in per.items() if not p["pdl_W2_over_best_stock_W2"] >= 0.95]
        v["iii"] = {"met": not f3, "fails": f3,
                    "pdl_W2_over_best_stock_W2": {lab: p["pdl_W2_over_best_stock_W2"] for lab, p in per.items()}}
    v["job"] = job
    v["nolora"] = ref
    # descriptive: the cold starts (load time, output checks, the dispatch's choices)
    cold = cold_rows(LB / a.cold, job)
    desc = {}
    for lab, r in sorted(cold.items()):
        c = r.get("check") or {}
        item = {"load_s": r.get("load_s"), "kv_tokens": r.get("kv_tokens")}
        for part in ("check", "check_strong", "check_b1"):
            if r.get(part):
                q = r[part]
                item[part] = {"ok": q.get("ok"), "tokens_identical": q.get("tokens_identical"),
                              "prompts_diverged": q.get("prompts_diverged"), "mean_abs_dlogp": q.get("mean_abs_dlogp"),
                              "missing_from_top5": q.get("missing_from_top5")}
        lay = r.get("layers") or {}
        if lay:
            item["layers"] = {k: lay.get(k) for k in ("fused_layers", "fallback_layers", "pdl_layers", "pdl_failures")}
            if lay.get("pdl_dispatch"):
                item["pdl_dispatch"] = lay["pdl_dispatch"]
        item["check_ok"] = c.get("ok") if c else None
        desc[lab] = item
    v["output_checks_ok"] = None if not desc else (
        all(x.get("check_ok") is not False for x in desc.values())
        and all(x.get("check_ok") for lab, x in desc.items() if lab.startswith("plugin-pdl+")))
    out.update({"settings": per, "verdict": v, "cold_starts": desc})
    for lab, p in per.items():
        print(lab, json.dumps({k: round(x, 4) for k, x in p.items()}))
    print("verdict:", json.dumps({k: v[k] for k in v if k in ("i", "ii", "iii")}, default=str)[:1500])
    print("output checks ok:", v["output_checks_ok"], "| cold starts:", len(desc))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0 if "status" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
