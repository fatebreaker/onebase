"""SHRINK-FLOOR (docs/REGISTRY.md, registered 2026-10-25; this scorer written before any data; mock review R1 v34 change 2): is the decode
path's batch-1 gain more than a slow shrink hidden? Rows: results/pdl_k2_floor_h200.jsonl (bench/pdl_k2_bench.py --stages floor, the
newest job): Qwen3-8B's four projections at one decode token, one adapter of rank 16, 64, 128, microseconds (CUDA graphs, L2 flushed).
Per rank, summed over the layer's four projections:
  pdl            the decode path (shrink launched first, our GEMM as its PDL dependent)
  serial_cublas  the down-projection as one cuBLAS bf16 GEMM, then our GEMM reading h: a vendor-fast shrink serialized
  nopdl          our shrink then our GEMM, serialized;  noshrink  our GEMM with h precomputed (the overlap's bound)
  shrink_cublas, shrink_only (ours), empty (an empty kernel: the launch floor)
  (i)  at rank 64: pdl < serial_cublas (the overlap beats serializing even a vendor-fast shrink)
Descriptive: the same at ranks 16 and 128; per projection the exposed shrink (pdl - noshrink), the GEMM's thread blocks against the
GPU's SMs.
    python3 bench/score_shrink_floor.py [--json results/shrink_floor_verdict.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
F = LB / "results/pdl_k2_floor_h200.jsonl"
KEYS = ("pdl", "serial_cublas", "pdl_nopdl", "pdl_noshrink", "shrink_cublas", "shrink_only", "empty")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--file", default=str(F), help="-FULL: results/pdl_k2_floor_full_h200.jsonl")
    a = ap.parse_args()
    f = Path(a.file)
    rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
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
                layer[rank] = {k: sum(x["t_" + k] for x in d.values()) for k in KEYS}
        out["layer_us"] = layer
        out["per_projection"] = {f"r{rank} {p}": {**{k: x.get("t_" + k) for k in KEYS},
                                                   "exposed": (x["t_pdl"] - x["t_pdl_noshrink"]) if isinstance(x.get("t_pdl_noshrink"), (int, float)) else None,
                                                   "k2_ctas": x.get("k2_ctas"), "shrink_ctas": x.get("shrink_ctas"), "n_sm": x.get("n_sm")}
                                 for rank, d in per.items() for p, x in d.items()}
        if 64 in layer:
            L = layer[64]
            out["i"] = {"met": L["pdl"] < L["serial_cublas"], "pdl_us": L["pdl"], "serial_cublas_us": L["serial_cublas"],
                        "ratio": L["pdl"] / L["serial_cublas"]}
    print(json.dumps(out, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
