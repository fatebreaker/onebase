"""STORED-VS-RANK and its extensions (docs/REGISTRY.md; descriptive): NVFP4 storage of a rank-64 adapter against its rank-r SVD
truncation (bench/svd_adapter.py) at equal or fewer bytes, for every adapter family evaluated: QMATH's Qwen3.5-9B math adapters
(S0G seeds 0-2; bf16 and NVFP4 bases), CODE-Q's code adapters (NVFP4 base, LiveCodeBench regraded), TIR-RL's tool-integrated math
adapters (bf16 and NVFP4 bases) and Q27-RL's Qwen3.8-27B math adapters (bf16 original and NVIDIA's NVFP4 release). The NVFP4-
stored rows are STORED-STRONG's, -2's and -27B's (results/std_bench/<adapter>_stored{,_nvrel}/); the truncations' rows are in
<adapter>_svd<r>{,_nvrel}/. Per adapter and base: accuracy, the gain retained R = (adapter on that base - that base) / (exact
adapter on bf16 - bf16 base), and the paired difference (rank-r minus NVFP4-stored) with a 95% bootstrap interval over problems
(bench/score_storerank.py's arithmetic, which this reproduces for rank 18 on math).
    python3 bench/score_storerank_all.py --rank 16 [--json results/std_bench/storerank16_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "bench"))
from score_storerank import acc, paired  # noqa: E402

RES = LB / "results/std_bench"
MATH = ("math500", "olympiad", "minerva")


def load(d: Path, benches: tuple) -> dict | None:
    """{(bench, id): correct} for one variant dir, or None if a file is missing; code reads the regraded verdicts."""
    out = {}
    for b in benches:
        f = d / ("lcb_regrade.jsonl" if b == "lcb" else f"{b}.jsonl")
        if not f.exists():
            return None
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if "correct" in r:
                out[(b, r["id"])] = float(r["correct"])
    return out


FAMILIES = {   # name: (adapter names, benches, {variant: dir suffix/variant}, bases compared)
    "math9": ([f"qwen9_dapo_S0G_s{k}" for k in range(3)], MATH,
              {"bf16_base": ("_stored", "bf16_base"), "bf16_exact": ("_stored", "bf16_exact"),
               "nvfp4_base": ("_stored", "nvfp4_base"), "nvfp4_exact": ("_stored", "nvfp4_exact"),
               "nvfp4_stored": ("_stored", "nvfp4_stored"), "bf16_stored": ("_stored", "bf16_stored"),
               "bf16_svd": ("_svd{r}", "bf16_stored"), "nvfp4_svd": ("_svd{r}", "nvfp4_stored")}, ("bf16", "nvfp4")),
    "code9": ([f"qwen9_code_s{k}" for k in range(3)], ("lcb",),
              {"bf16_base": ("_stored", "bf16_base"), "bf16_exact": ("_stored", "bf16_exact"),
               "nvfp4_base": ("_stored", "nvfp4_base"), "nvfp4_exact": ("_stored", "nvfp4_exact"),
               "nvfp4_stored": ("_stored", "nvfp4_stored"), "nvfp4_svd": ("_svd{r}", "nvfp4_stored")}, ("nvfp4",)),
    "tir9": ([f"qwen9_tir_s{k}" for k in range(3)], MATH,
             {"bf16_base": ("_stored", "bf16_base"), "bf16_exact": ("_stored", "bf16_exact"),
              "nvfp4_base": ("_stored", "nvfp4_base"), "nvfp4_exact": ("_stored", "nvfp4_exact"),
              "nvfp4_stored": ("_stored", "nvfp4_stored"), "bf16_svd": ("_svd{r}", "bf16_stored"),
              "nvfp4_svd": ("_svd{r}", "nvfp4_stored")}, ("nvfp4",)),
    "math27": ([f"qwen27_dapo_s{k}" for k in range(3)], MATH,
               {"bf16_base": ("_stored", "bf16_base"), "bf16_exact": ("_stored", "bf16_exact"),
                "nvfp4_base": ("_stored_nvrel", "bf16_base"), "nvfp4_exact": ("_stored_nvrel", "bf16_exact"),
                "nvfp4_stored": ("_stored_nvrel", "bf16_stored"), "bf16_svd": ("_svd{r}", "bf16_stored"),
                "nvfp4_svd": ("_svd{r}_nvrel", "bf16_stored")}, ("nvfp4",)),
}


def family(name: str, r: int) -> dict:
    names, benches, spec, bases = FAMILIES[name]
    out = {}
    for n in names:
        v = {k: load(RES / f"{n}{suf.format(r=r)}" / var, benches) for k, (suf, var) in spec.items()}
        gain = acc(v["bf16_exact"]) - acc(v["bf16_base"]) if v["bf16_exact"] and v["bf16_base"] else None
        row = {"bf16_gain": gain, "missing": [k for k, x in v.items() if x is None]}
        for base in ("bf16", "nvfp4"):
            b0 = v.get(f"{base}_base")
            for kind in ("exact", "stored", "svd"):
                x = v.get(f"{base}_{kind}")
                if x is not None and b0 is not None:
                    row[f"{base}_{kind}_acc"] = acc(x)
                    row[f"{base}_{kind}_R"] = (acc(x) - acc(b0)) / gain if gain else None
            if base in bases and v.get(f"{base}_svd") is not None and v.get(f"{base}_stored") is not None:
                row[f"{base}_svd_minus_stored"] = paired(v[f"{base}_svd"], v[f"{base}_stored"])
        st = Path("$SCRATCH/std_bench/adapters") / f"{n}_svd{r}" / "svd_stats.json"
        if st.exists():
            row["kept_frac_mean"] = json.loads(st.read_text())["kept_frac_mean"]
        out[n] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out = {f: family(f, a.rank) for f in FAMILIES}
    for f, rows in out.items():
        for n, row in rows.items():
            show = {k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items() if not isinstance(x, dict)}
            diffs = {k: (round(x["diff"], 2), [round(c, 2) for c in x["ci95"]]) for k, x in row.items() if isinstance(x, dict)}
            print(f, n, json.dumps(show), json.dumps(diffs))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
