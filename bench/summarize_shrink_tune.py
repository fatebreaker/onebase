"""Summarize results/shrink_tune.jsonl (CPU): per GPU and batch size, the shrink configuration (tiles and split) that is
within 5% of the best on the most (projection, rank) cases, its geometric-mean time against the stock path, and the
default's. Used to set the shrink part of fp8_lora_vllm.pick_cfg per architecture.
    python bench/summarize_shrink_tune.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "shrink_tune.jsonl"


def key(c: dict) -> tuple:
    return (c["BLOCK_M"], c["BLOCK_SR"], c["BLOCK_K"], c["num_warps"], c["num_stages"], c.get("split"))


def main() -> int:
    rows = [json.loads(x) for x in R.read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["gpu"], r.get("op", "fp8"), r.get("fmt", "bf16"), r["M"])].append(r)
    for (gpu, op, fmt, M), rs in sorted(by.items()):
        votes = Counter()
        for r in rs:
            best = r["t_best"]
            for t, c in r["top3"]:
                if t <= 1.05 * best:
                    votes[key(c)] += 1
        geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))   # noqa: E731
        best_ratio = geo([r["t_best"] / r["t_stock"] for r in rs])
        dflt = [r["t_default"] / r["t_stock"] for r in rs if r.get("t_default")]
        ship = [r["t_shipped"] / r["t_stock"] for r in rs if r.get("t_shipped")]
        print(f"{gpu[:24]:24s} {op:5s} {fmt:4s} M={M:5d} n={len(rs)}  best/stock {best_ratio:.2f}  default/stock "
              f"{geo(dflt) if dflt else float('nan'):.2f}  shipped/stock {geo(ship) if ship else float('nan'):.2f}  top configs (BM,SR,BK,warps,stages,split): {votes.most_common(3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
