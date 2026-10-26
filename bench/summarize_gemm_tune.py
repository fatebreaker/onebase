"""Summarize results/gemm_tune.jsonl (CPU): per GPU and batch size, the fused path's time with pick_cfg's default
GEMM tiles and with the best searched tiles, against the stock and hybrid paths (geometric means over projections and
ranks), and the tiles that are within 3% of the best in the most cases (candidates for a per-architecture pick_cfg).
    python bench/summarize_gemm_tune.py [--gpu H100]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "gemm_tune.jsonl"


def key(c: dict) -> tuple:
    return (c["BLOCK_M"], c["BLOCK_N"], c["BLOCK_K"], c["num_warps"], c["num_stages"])


def geo(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="")
    a = ap.parse_args()
    rows = [json.loads(x) for x in R.read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows:
        if a.gpu in r["gpu"]:
            by[(r["gpu"], r["M"])].append(r)
    print("per M: geometric means over (projection, rank) of time / stock; fused = our GEMM + LoRA epilogue")
    for (gpu, M), rs in sorted(by.items()):
        votes = Counter()
        for r in rs:
            for t, c in r["top5"]:
                if t <= 1.03 * r["t_best"]:
                    votes[key(c)] += 1
        print(f"{gpu[:24]:24s} M={M:4d} n={len(rs)}  hybrid {geo(r['t_hybrid'] / r['t_stock'] for r in rs):.2f}  "
              f"fused default {geo(r['t_default'] / r['t_stock'] for r in rs if r['t_default']):.2f}  "
              f"fused best {geo(r['t_best'] / r['t_stock'] for r in rs):.2f}  "
              f"(vLLM no-LoRA {geo(r['t_ref'] / r['t_stock'] for r in rs):.2f})  "
              f"tiles (BM,BN,BK,warps,stages) within 3%: {votes.most_common(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
