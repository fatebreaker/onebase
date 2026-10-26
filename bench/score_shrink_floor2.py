"""SHRINK-FLOOR amendment -FLOOR2 (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v37,
change 2): the floor stage at the full tile sweep with the serialized empty-kernel arm (LOWBIT_FLOOR_EMPTY=1: an empty kernel, then the
same GEMM reading h: the floor of any serialized shrink). Rows: results/pdl_k2_floor2_h200.jsonl (the newest job). Per rank, summed
over Qwen3-8B's four projections at one decode token:
  (i) at rank 64: pdl < serial_empty (the overlap beats even a serialized shrink that does no work)
Descriptive: ranks 16 and 128; serial_empty / pdl (the overlap's own gain over the floor), serial_cublas / pdl; per projection the
exposed shrink and the GEMM's thread blocks.
    python3 bench/score_shrink_floor2.py [--json results/shrink_floor2_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/pdl_k2_floor2_h200.jsonl"
KEYS = ("pdl", "serial_empty", "serial_cublas", "pdl_nopdl", "pdl_noshrink", "shrink_cublas", "shrink_only", "empty")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rs = [json.loads(x) for x in F.read_text().splitlines() if x.strip()] if F.exists() else []
    rs = [r for r in rs if r.get("stage") == "floor"]
    out: dict = {}
    if rs:
        job = max(str(r.get("job", "")) for r in rs)
        rs = [r for r in rs if str(r.get("job", "")) == job]
        out["job"] = job
        per: dict = {}
        for r in rs:
            per.setdefault(r["r"], {})[r["proj"]] = r
        layer = {}
        for rank, d in sorted(per.items()):
            if len(d) == 4 and all(all(isinstance(x.get("t_" + k), (int, float)) for k in KEYS) for x in d.values()):
                L = {k: sum(x["t_" + k] for x in d.values()) for k in KEYS}
                L["empty_over_pdl"] = L["serial_empty"] / L["pdl"]
                L["cublas_over_pdl"] = L["serial_cublas"] / L["pdl"]
                layer[rank] = L
        out["layer_us"] = layer
        out["per_projection"] = {f"r{rank} {p}": {"pdl": x.get("t_pdl"), "serial_empty": x.get("t_serial_empty"),
                                                   "exposed": x["t_pdl"] - x["t_pdl_noshrink"] if isinstance(x.get("t_pdl_noshrink"), (int, float)) else None,
                                                   "k2_ctas": x.get("k2_ctas"), "n_sm": x.get("n_sm")}
                                 for rank, d in per.items() for p, x in d.items()}
        if 64 in layer:
            L = layer[64]
            out["i"] = {"met": L["pdl"] < L["serial_empty"], "pdl_us": L["pdl"], "serial_empty_us": L["serial_empty"],
                        "ratio": L["serial_empty"] / L["pdl"]}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
