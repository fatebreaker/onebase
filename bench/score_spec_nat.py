"""SPEC-DECODE-NAT-H200 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v37, missing 3):
SPEC-DECODE-H200 on WildChat's first turns (mlsys jobs/pod_runs/spec_nat_h200_pod.sh -> results/spec_nat_h200.jsonl); means of the three
measured starts per mode.
  (i)  N1: TPOT(plugin-pdl-spec4) < TPOT(stock-tuned-spec4)
  (ii) N4 and N8: the same at four and eight concurrent requests
Descriptive: accepted / draft tokens and tokens per verify step; speculation's speedup for each engine (spec0 over spec4).
    python3 bench/score_spec_nat.py [--json results/spec_nat_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/spec_nat_h200.jsonl"
BS = (1, 4, 8)


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
        for b in BS:
            key = f"N{b}_tpot_ms"
            if all(key in x for x in v.values()):
                m[k][key] = sum(x[key] for x in v.values()) / 3
                sp = [x.get(f"N{b}_spec") or {} for x in v.values()]
                acc, dr, nd = (sum(s.get(f) or 0 for s in sp) for f in ("accepted_tokens", "draft_tokens", "drafts"))
                m[k][f"N{b}_accept_rate"] = round(acc / dr, 3) if dr else None
                m[k][f"N{b}_tokens_per_step"] = round(1 + acc / nd, 3) if nd else None
    out: dict = {"means": m}
    have = lambda k, b: f"N{b}_tpot_ms" in m.get(k, {})  # noqa: E731
    if have("plugin-pdl-spec4", 1) and have("stock-tuned-spec4", 1):
        p, s = m["plugin-pdl-spec4"]["N1_tpot_ms"], m["stock-tuned-spec4"]["N1_tpot_ms"]
        out["i"] = {"met": p < s, "plugin_ms": round(p, 3), "vllm_ms": round(s, 3), "ratio": round(p / s, 3)}
    if all(have(k, b) for k in ("plugin-pdl-spec4", "stock-tuned-spec4") for b in (4, 8)):
        rs = {b: m["plugin-pdl-spec4"][f"N{b}_tpot_ms"] / m["stock-tuned-spec4"][f"N{b}_tpot_ms"] for b in (4, 8)}
        out["ii"] = {"met": all(r < 1 for r in rs.values()), "ratios": {str(b): round(r, 3) for b, r in rs.items()}}
    for eng in ("plugin-pdl", "stock-tuned"):
        if all(have(f"{eng}-spec{s}", b) for s in (0, 4) for b in BS):
            out[f"{eng}_spec_speedup"] = {str(b): round(m[f"{eng}-spec0"][f"N{b}_tpot_ms"] / m[f"{eng}-spec4"][f"N{b}_tpot_ms"], 3) for b in BS}
    out["complete"] = "i" in out and "ii" in out
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
