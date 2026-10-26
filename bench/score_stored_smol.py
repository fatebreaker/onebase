"""STORED-SMOL (docs/REGISTRY.md): SmolLM2 GRPO adapters stored in NVFP4 / FP8 and served exactly on the matching
weights-only base (std_bench.py variants nvfp4_stored / fp8_stored in each adapter's own results/std_bench/<adapter>/).
R = (acc - acc_<fmt>_base) / (acc_bf16_exact - acc_bf16_base), paired bootstrap 10,000; judged if the bf16 gain >= 3.
  (i) NVFP4-stored R >= 0.8, (ii) FP8-stored R >= 0.8, each on every judged adapter whose stored variant is evaluated.
Descriptive: McNemar of each stored variant against the bf16 exact adapter on the same base.
    python3 bench/score_stored_smol.py [--seeds 0 1 2 3 4 5] [--json results/std_bench/stored_smol_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402


def load(f: Path) -> dict | None:
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows, ok = {}, {"nvfp4": [], "fp8": []}
    for s in a.seeds:
        n = f"smol_gsm8k_rlb_s{s}"
        v = {k: load(RES / n / k / "gsm8k.jsonl") for k in ("bf16_base", "bf16_exact", "nvfp4_base", "nvfp4_exact",
                                                            "nvfp4_stored", "fp8_base", "fp8_exact", "fp8_stored")}
        if v["bf16_base"] is None or v["bf16_exact"] is None:
            continue
        gain = 100 * (sum(v["bf16_exact"].values()) / len(v["bf16_exact"]) - sum(v["bf16_base"].values()) / len(v["bf16_base"]))
        row = {"gain_pts": gain, "judged": gain >= 3}
        for fmt in ("nvfp4", "fp8"):
            st, ex = v[f"{fmt}_stored"], v[f"{fmt}_exact"]
            if st is None or ex is None:
                continue
            r = rr.retention(v["bf16_base"], v["bf16_exact"], v[f"{fmt}_base"], st, n_boot=10000)
            re_ = rr.retention(v["bf16_base"], v["bf16_exact"], v[f"{fmt}_base"], ex, n_boot=10000)
            ids = sorted(set(st) & set(ex))
            w, l, p = rr.mcnemar([st[i] for i in ids], [ex[i] for i in ids])
            row[fmt] = {"stored": {"R": r["R"], "R_ci95": r["R_ci95"]}, "exact": {"R": re_["R"], "R_ci95": re_["R_ci95"]},
                        "stored_vs_exact": {"wins": w, "losses": l, "p": p}}
            if row["judged"]:
                ok[fmt].append(r["R"] >= 0.8)
            print(f"{n} ({gain:+.1f}) {fmt}: stored R {r['R']:.2f} [{r['R_ci95'][0]:.2f}, {r['R_ci95'][1]:.2f}], exact "
                  f"{re_['R']:.2f}; stored vs exact {w}:{l} p {p:.2g}")
        rows[n] = row
    verdict = {k: (f"{'MET' if all(x) else 'NOT MET'} ({sum(x)} of {len(x)} judged)" if x else "no data")
               for k, x in (("i", ok["nvfp4"]), ("ii", ok["fp8"]))}
    print("verdict:", verdict)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
