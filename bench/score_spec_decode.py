"""SPEC-DECODE-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock reviews R1 v34 M2, R3 v35
M1 / missing 5): the decode path under vLLM's n-gram speculative decoding (k = 4) on one H200 (mlsys jobs/pod_runs/
spec_decode_h200_pod.sh -> results/spec_decode_h200.jsonl); means of the three measured starts per mode.
  (i)  S1: TPOT(plugin-pdl-spec4) < min(TPOT of stock-tuned-spec4, stock-dual-tuned-spec4)
  (ii) S4: the same at four requests
Descriptive: each mode's accepted / draft tokens and tokens per verify step; the plugin's ratio to vLLM's best with and without
speculation (spec0 modes); each LoRA path's cost over nolora-spec4; whether the S1 first tokens agree across modes.
    python3 bench/score_spec_decode.py [--json results/spec_decode_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/spec_decode_h200.jsonl"
VLLM = ("stock-tuned-spec4", "stock-dual-tuned-spec4")


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
    m: dict = {}
    for k, v in got.items():
        if len(v) != 3:
            continue
        m[k] = {}
        for b in (1, 4):
            key = f"S{b}_tpot_ms"
            if all(key in x for x in v.values()):
                m[k][key] = sum(x[key] for x in v.values()) / 3
                sp = [x.get(f"S{b}_spec") or {} for x in v.values()]
                acc = sum(s.get("accepted_tokens") or 0 for s in sp)
                dr = sum(s.get("draft_tokens") or 0 for s in sp)
                nd = sum(s.get("drafts") or 0 for s in sp)
                m[k][f"S{b}_accept_rate"] = round(acc / dr, 3) if dr else None
                m[k][f"S{b}_tokens_per_step"] = round(1 + acc / nd, 3) if nd else None
        m[k]["S1_first_tokens"] = next(iter(v.values())).get("S1_first_tokens")
    out: dict = {"means": m}
    for tag, b in (("i", 1), ("ii", 4)):
        key = f"S{b}_tpot_ms"
        if all(key in m.get(x, {}) for x in VLLM + ("plugin-pdl-spec4",)):
            best = min(VLLM, key=lambda x: m[x][key])
            p = m["plugin-pdl-spec4"][key]
            out[tag] = {"met": p < m[best][key], "plugin_ms": round(p, 3), "vllm_best": best, "vllm_best_ms": round(m[best][key], 3),
                        "ratio": round(p / m[best][key], 3)}
    for b in (1, 4):
        key = f"S{b}_tpot_ms"
        if all(key in m.get(x, {}) for x in ("plugin-pdl-spec0", "stock-tuned-spec0")):
            out[f"S{b}_no_spec_ratio"] = round(m["plugin-pdl-spec0"][key] / m["stock-tuned-spec0"][key], 3)
        if "nolora-spec4" in m and key in m["nolora-spec4"]:
            out[f"S{b}_cost_over_nolora"] = {x: round(m[x][key] / m["nolora-spec4"][key] - 1, 3) for x in m
                                             if x.endswith("spec4") and x != "nolora-spec4" and key in m[x]}
    ft = {k: v.get("S1_first_tokens") for k, v in m.items() if v.get("S1_first_tokens")}
    if ft:
        ref = ft.get("stock-tuned-spec4") or next(iter(ft.values()))
        out["S1_first_tokens_agree"] = {k: v == ref for k, v in ft.items()}
    out["complete"] = "i" in out and "ii" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
