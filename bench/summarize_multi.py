"""Summarize a multi-adapter kernel benchmark (bench/multi_bench.py PLUGIN=1; CPU, seconds): per (rank, M, n), the
LoRA overhead of a decoder layer (the four Llama-3.1-8B projections summed) over vLLM's no-adapter quantize + GEMM,
for every variant; parity failures; and at M = 64 the share of peak DRAM bandwidth each path reaches (bytes = FP8
base + the n adapters' A and B in their storage format + x in + y out).
    python bench/summarize_multi.py <job> [--model llama31_8b] [--bw 864]     (reads results/multi_kernel_plugin.jsonl)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

SHAPES = {   # (K, N, slices) per projection
    "llama31_8b": {"qkv": (4096, 6144, 3), "o": (4096, 4096, 1), "gate_up": (4096, 28672, 2), "down": (14336, 4096, 1)},
    "qwen3_8b": {"qkv": (4096, 6144, 3), "o": (4096, 4096, 1), "gate_up": (4096, 24576, 2), "down": (12288, 4096, 1)}}
BYTES_PER_W = {"": 2.0, "8": 1.0, "4": 0.5 + 1 / 16}      # adapter storage: bf16, FP8 (+ per-row scales), NVFP4
PEAK = {"L40S": 864, "H100": 3350, "RTX PRO 6000": 1792}   # GB/s (H100 SXM; RTX PRO 6000 full GPU, not a MIG slice)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job")
    ap.add_argument("--model", default="llama31_8b")
    ap.add_argument("--results", default="results/multi_kernel_plugin.jsonl")
    ap.add_argument("--bw", type=float, default=0.0, help="peak DRAM bandwidth in GB/s (default: from the GPU name)")
    a = ap.parse_args()
    rows = [r for r in map(json.loads, open(a.results)) if r["job"] == a.job and r["model"] == a.model]
    gpu = rows[0]["gpu"] if rows else ""
    bw = a.bw or next((v for k, v in PEAK.items() if k in gpu), 0.0)
    bad = [(r["proj"], r["r"], r["M"], r["n"], k) for r in rows for k, v in r.get("ok", {}).items() if not v]
    print(f"{gpu.strip()[:40]} {a.model}  rows {len(rows)}  parity failures: {bad if bad else 'none'}")
    agg: dict = defaultdict(lambda: defaultdict(float))
    cnt: dict = defaultdict(int)
    for r in rows:
        k = (r["r"], r["M"], r["n"])
        cnt[k] += 1
        for v, t in r["t_us"].items():
            agg[k][v] += t
    order = ("stock", "stock8", "plugin", "pluginh", "pluginq", "plugin8", "plugin8h", "plugin8q", "plugin4", "plugin4h",
             "plugin4q")
    variants = sorted({v for d in agg.values() for v in d if v != "ref"},
                      key=lambda v: order.index(v) if v in order else len(order))
    print("layer overhead over vLLM's no-adapter path (%), four projections summed:")
    print("  (r, M, n)       " + " ".join(f"{v:>8s}" for v in variants))
    for k in sorted(agg):
        if cnt[k] < 4:
            continue
        ref = agg[k]["ref"]
        print(f"  {str(k):15s} " + " ".join(f"{100 * (agg[k][v] / ref - 1):8.1f}" if v in agg[k] else f"{'-':>8s}"
                                          for v in variants))
    if bw:
        print(f"share of peak DRAM bandwidth ({bw:.0f} GB/s) at M = 64, per projection:")
        for r in rows:
            if r["M"] != 64:
                continue
            K, N, S = SHAPES[a.model][r["proj"]]
            base, io = N * K * 1.0, 64 * K * 2 + 64 * N * 2
            ad = r["n"] * r["r"] * (S * K + N)
            out = []
            for v, t in r["t_us"].items():
                fmt = "8" if "8" in v else "4" if "4" in v else ""
                b = base + io + (0 if v == "ref" else ad * BYTES_PER_W[fmt])
                out.append(f"{v} {b / (t * 1e-6) / (bw * 1e9):.2f}")
            print(f"  {r['proj']:7s} r{r['r']:<3d} n{r['n']:<3d} " + "  ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
