"""STORED-VS-RANK (docs/REGISTRY.md; descriptive): NVFP4 storage against rank truncation at equal adapter bytes. For QMATH's
Qwen3.5-9B S0G adapters (seeds 0-2): the NVFP4-stored rank-64 adapter (STORED-STRONG's evaluations, results/std_bench/
<adapter>_stored/<base>_stored) and the bf16 rank-18 SVD truncation (bench/svd_adapter.py; results/std_bench/<adapter>_svd18/
<base>_stored), each on the bf16 base and on the NVFP4 base, MATH-500 + OlympiadBench + Minerva pooled, greedy, A100.
Per adapter and base: accuracy, the gain retained R = (adapter on that base - base) / (exact adapter on bf16 - bf16 base), and
the paired difference (rank-18 minus NVFP4-stored) with a 95% bootstrap interval over problems.
    python3 bench/score_storerank.py [--json results/std_bench/storerank_verdict.json]"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
BENCHES = ("math500", "olympiad", "minerva")


def load(d: Path) -> dict | None:
    """{(bench, id): correct} for one variant dir, or None if any bench is missing."""
    out = {}
    for b in BENCHES:
        f = d / f"{b}.jsonl"
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "correct" in r:
                out[(b, r["id"])] = float(r["correct"])
    return out


def acc(x: dict) -> float:
    return 100 * sum(x.values()) / len(x)


def paired(a: dict, b: dict, n: int = 2000, seed: int = 0) -> dict:
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    m = len(d)
    vals = sorted(100 * sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"diff": 100 * sum(d) / m, "ci95": [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]], "problems": m}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {}
    for k in (0, 1, 2):
        name = f"qwen9_dapo_S0G_s{k}"
        st, sv = RES / f"{name}_stored", RES / f"{name}_svd18"
        v = {x: load(st / x) for x in ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact", "nvfp4_stored", "bf16_stored")}
        v.update({"bf16_svd18": load(sv / "bf16_stored"), "nvfp4_svd18": load(sv / "nvfp4_stored")})
        if any(x is None for x in v.values()):
            out[name] = {"status": "incomplete", "missing": [x for x, y in v.items() if y is None]}
            continue
        gain = acc(v["bf16_exact"]) - acc(v["bf16_base"])
        row = {"bf16_gain": gain}
        for base in ("bf16", "nvfp4"):
            b0 = acc(v[f"{base}_base"])
            for kind in ("exact", "stored", "svd18"):
                row[f"{base}_{kind}_acc"] = acc(v[f"{base}_{kind}"])
                row[f"{base}_{kind}_R"] = (acc(v[f"{base}_{kind}"]) - b0) / gain if gain else None
            row[f"{base}_svd18_minus_stored"] = paired(v[f"{base}_svd18"], v[f"{base}_stored"])
        out[name] = row
        print(name, json.dumps({x: (round(y, 3) if isinstance(y, float) else y) for x, y in row.items()}))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
