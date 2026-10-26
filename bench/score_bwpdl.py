"""BLACKWELL-PDL-E2E (docs/REGISTRY.md, registered 2026-10-22 00:13 UTC before any data; this scorer written before any
data): the plugin with the opt-in NVFP4 PDL decode path (plugin-pdl) end to end on one RTX PRO 6000 Blackwell MIG 2g.48gb
slice against vLLM's LoRA path with its default, slice-tuned and dual-stream kernels and the plugin without the PDL path,
from results/e2e_bwpdl.jsonl (jobs/e2e_bwpdl.sbatch; labels <mode>+r<r>:n<n>#<start> and nolora#<start>). Each label's
newest complete starts: per configuration the mean of its newest three measured starts holding both W1 and W2 (by time,
from any run of the job script: a rerun's starts supersede earlier ones label by label). Settings with LoRA (rank,
adapters): (64, 1), (16, 32), (64, 32).
  (i)   plugin-pdl's W1 (batch-1 time per output token) below stock-tuned's and below stock-dual's at every setting with LoRA
  (ii)  plugin-pdl's W1 at 64:1 at most 1.10x the no-LoRA engine's
  (iii) plugin-pdl's W2 (throughput) at least 0.95x the better (higher) of stock-tuned's and stock-dual's at every setting
        with LoRA
Each met at every setting or not met (the settings where it fails listed); incomplete while a configuration lacks three
measured starts. Validity (registered): at every setting with LoRA, plugin-pdl's cold-start output check against stock's
(results/e2e_bwpdl_cold.jsonl, bench/e2e_outcheck.compare on 16 prompts: the mean |dlog p| of stock's tokens up to each
prompt's first divergence, and the number of diverged prompts) has a mean |dlog p| at most 1.5x the plugin mode's and at
most 3 more diverged prompts than the plugin mode's; otherwise, or where a check is missing, the verdicts are reported as
unverified. Per label the newest cold start holding a check is used. Descriptive: W1 and W2 of every mode and setting and
their change against the no-LoRA engine, plugin-pdl over the plugin; every mode's output checks (also on strong adapters
and at batch 1); the dispatch's choices per layer and bucket (the plugin modes' cold starts); cold-start (engine load)
times; the jobs and times of the starts used.
    python3 bench/score_bwpdl.py [--json results/std_bench/bwpdl_verdict.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
MODES = ("stock", "stock-tuned", "stock-dual", "plugin", "plugin-pdl")
SETTINGS = ((64, 1), (16, 32), (64, 32))
REPS = 3


def key_of(label: str):
    """'<mode>+r<r>:n<n>#<start>' -> (mode, (r, n)); 'nolora#<start>' -> ('nolora', None); anything else None."""
    lab = label.rsplit("#", 1)[0]
    if lab == "nolora":
        return ("nolora", None)
    if "+r" in lab and ":n" in lab:
        m, s = lab.split("+", 1)
        rk, nk = s.split(":", 1)
        try:
            return (m, (int(rk[1:]), int(nk[1:])))
        except ValueError:
            return None
    return None


def lab_of(k) -> str:
    return "nolora" if k[1] is None else f"{k[0]}+r{k[1][0]}:n{k[1][1]}"


def read(path: Path) -> list:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def load_newest(path: Path, reps: int = REPS) -> dict:
    """{(mode, (r, n)) or ('nolora', None): the label's newest `reps` complete measured starts}, oldest first."""
    by: dict = {}
    for r in sorted(read(path), key=lambda r: r.get("time", "")):
        k = key_of(r.get("label", ""))
        if k is None or r.get("label", "").endswith("#cold") or "W1_tpot_ms" not in r or "W2_gen_tok_s" not in r:
            continue
        by.setdefault(k, []).append(r)
    return {k: v[-reps:] for k, v in by.items()}


def cold_newest(path: Path) -> dict:
    """{label: its newest cold start holding an output check, else its newest cold start}."""
    best: dict = {}
    for r in sorted(read(path), key=lambda r: r.get("time", "")):
        if not r.get("label", "").endswith("#cold"):
            continue
        lab = r["label"].rsplit("#", 1)[0]
        if r.get("check") or not (best.get(lab) or {}).get("check"):
            best[lab] = r
    return best


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--data", default="results/e2e_bwpdl.jsonl")
    ap.add_argument("--cold", default="results/e2e_bwpdl_cold.jsonl")
    a = ap.parse_args()
    need = [(m, s) for m in MODES for s in SETTINGS] + [("nolora", None)]
    d = load_newest(LB / a.data)
    miss = [k for k in need if len(d.get(k, [])) < REPS]
    out: dict = {"data": a.data, "cold": a.cold}
    if miss:
        out["status"] = "incomplete"
        out["missing"] = [lab_of(k) for k in miss]
        print(f"incomplete: {len(miss)} configurations without {REPS} complete measured starts, e.g. {out['missing'][0]}")
    e = {k: {"W1": mean([r["W1_tpot_ms"] for r in v]), "W2": mean([r["W2_gen_tok_s"] for r in v])}
         for k, v in d.items() if len(v) >= REPS}
    used = {lab_of(k): [{"job": r.get("job"), "time": r.get("time"), "W1": r["W1_tpot_ms"], "W2": r["W2_gen_tok_s"]}
                        for r in v] for k, v in sorted(d.items(), key=lambda kv: lab_of(kv[0])) if k in [*need]}
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
    # validity: plugin-pdl's output check against the plugin's (the calibration), at every setting with LoRA
    cold = cold_newest(LB / a.cold)
    val: dict = {}
    for s in SETTINGS:
        lab = f"{s[0]}:{s[1]}"
        cp = (cold.get(f"plugin-pdl+r{s[0]}:n{s[1]}") or {}).get("check")
        cq = (cold.get(f"plugin+r{s[0]}:n{s[1]}") or {}).get("check")
        if not cp or not cq:
            val[lab] = {"ok": False, "why": "missing check: " + ", ".join(
                m for m, c in (("plugin-pdl", cp), ("plugin", cq)) if not c)}
            continue
        dp, dq = float(cp["mean_abs_dlogp"]), float(cq["mean_abs_dlogp"])
        np_, nq = int(cp["prompts_diverged"]), int(cq["prompts_diverged"])
        val[lab] = {"ok": dp <= 1.5 * dq and np_ <= nq + 3, "pdl_mean_abs_dlogp": dp, "plugin_mean_abs_dlogp": dq,
                    "dlogp_ratio": dp / dq if dq > 0 else (1.0 if dp == 0 else float("inf")),
                    "pdl_prompts_diverged": np_, "plugin_prompts_diverged": nq}
    valid = all(x["ok"] for x in val.values())
    v["validity"] = {"valid": valid, "settings": val}
    v["status"] = ("incomplete" if "status" in out else "verified" if valid else "unverified")
    v["nolora"] = ref
    # descriptive: every mode's cold start (load time, output checks, the dispatch's choices)
    desc = {}
    for lab, r in sorted(cold.items()):
        item = {"job": r.get("job"), "time": r.get("time"), "load_s": r.get("load_s"), "kv_tokens": r.get("kv_tokens")}
        for part in ("check", "check_strong", "check_b1"):
            if r.get(part):
                q = r[part]
                item[part] = {"ok": q.get("ok"), "tokens_identical": q.get("tokens_identical"),
                              "prompts_diverged": q.get("prompts_diverged"), "mean_abs_dlogp": q.get("mean_abs_dlogp"),
                              "missing_from_top5": q.get("missing_from_top5")}
        lay = r.get("layers") or {}
        if lay:
            item["layers"] = {k: lay.get(k) for k in ("fused_layers", "fallback_layers", "pdl_layers", "pdl_failures",
                                                      "op_called")}
            if lay.get("pdl_dispatch"):
                item["pdl_dispatch"] = lay["pdl_dispatch"]
        desc[lab] = item
    out.update({"settings": per, "verdict": v, "cold_starts": desc, "starts_used": used})
    for lab, p in per.items():
        print(lab, json.dumps({k: round(x, 4) for k, x in p.items()}))
    print("verdict:", json.dumps({k: v[k] for k in v if k in ("i", "ii", "iii")}, default=str)[:1500])
    print("validity:", json.dumps(v["validity"], default=str)[:800])
    print("status:", v["status"], "| cold starts:", len(desc))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0 if "status" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
