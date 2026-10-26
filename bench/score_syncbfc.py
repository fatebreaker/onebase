"""SYNC-BF16-C (docs/REGISTRY.md): SmolLM2 GRPO, exact-sync vs bf16 merge-sync pairs at seeds 6, 7, 8 (train/grpo_lora4.py;
logs/lb_syncbfc_smol_gsm8k_bfc_<exact|bf16merge>_s<k>.out, results/std_bench/smol_gsm8k_bfc_<sync>_s<k>/bf16_exact/gsm8k.jsonl).
  (i)  the merge-sync's final GSM8K accuracy is at least 10 points below its exact twin's (McNemar p <= 0.05) in >= 2 of 3
  (ii) the merge-sync's mean trainer-engine mismatch over steps 1-100 is at least twice its twin's, in every seed
Descriptive: delivered share (syncs 1-10, 191-200), reward and length trajectories, the first step at which the 10-step
mean reward falls below half its running maximum.
    python3 bench/score_syncbfc.py [--json results/std_bench/syncbfc_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
RES = LB / "results/std_bench"
sys.path.insert(0, "$ROUNDING/analysis")
import read_retention as rr  # noqa: E402


def log(sync: str, s: int) -> list[dict]:
    f = LB / f"logs/lb_syncbfc_smol_gsm8k_bfc_{sync}_s{s}.out"
    return [json.loads(x) for x in f.read_text().splitlines() if x.startswith('{"step"')] if f.exists() else []


def acc(sync: str, s: int) -> dict | None:
    f = RES / f"smol_gsm8k_bfc_{sync}_s{s}/bf16_exact/gsm8k.jsonl"
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def collapse_step(rows: list[dict]) -> int | None:
    best = 0.0
    for t in range(10, len(rows) + 1):
        m = mean([r["reward"] for r in rows[t - 10:t]])
        best = max(best, m)
        if m < 0.5 * best:
            return rows[t - 1]["step"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    out, i_hits, ii_all, n = {}, 0, True, 0
    for s in a.seeds:
        ex, mg = log("exact", s), log("bf16merge", s)
        ae, am = acc("exact", s), acc("bf16merge", s)
        if not ex or not mg or ex[-1]["step"] < 200 or mg[-1]["step"] < 200 or ae is None or am is None:
            print(f"seed {s}: not complete")
            continue
        n += 1
        ids = sorted(set(ae) & set(am))
        pe, pm = 100 * sum(ae[i] for i in ids) / len(ids), 100 * sum(am[i] for i in ids) / len(ids)
        w, l, p = rr.mcnemar([ae[i] for i in ids], [am[i] for i in ids])
        mm_e = mean([r.get("mismatch") for r in ex if r["step"] <= 100])
        mm_m = mean([r.get("mismatch") for r in mg if r["step"] <= 100])
        hit_i = pe - pm >= 10 and w > l and p <= 0.05
        hit_ii = mm_m >= 2 * mm_e
        i_hits += hit_i
        ii_all &= hit_ii
        out[str(s)] = {"acc_exact": pe, "acc_merge": pm, "exact_vs_merge": [w, l, p], "i": hit_i,
                       "mismatch_1_100": {"exact": mm_e, "merge": mm_m}, "ii": hit_ii,
                       "delivered": [mean([r["sync_gain"] for r in mg if r["step"] <= 10]),
                                     mean([r["sync_gain"] for r in mg if r["step"] > 190])],
                       "reward_191_200": {"exact": mean([r["reward"] for r in ex if r["step"] > 190]),
                                          "merge": mean([r["reward"] for r in mg if r["step"] > 190])},
                       "len_191_200": {"exact": mean([r["len"] for r in ex if r["step"] > 190]),
                                       "merge": mean([r["len"] for r in mg if r["step"] > 190])},
                       "collapse_step": {"exact": collapse_step(ex), "merge": collapse_step(mg)}}
        print(f"seed {s}: GSM8K exact {pe:.1f} vs merge {pm:.1f} ({w}:{l}, p {p:.2g}); mismatch 1-100 {mm_e:.4f} vs {mm_m:.4f}; "
              f"delivered {out[str(s)]['delivered'][0]:.3f} -> {out[str(s)]['delivered'][1]:.3f}; collapse "
              f"{out[str(s)]['collapse_step']}")
    if n == len(a.seeds):
        v = {"i": i_hits >= 2, "i_hits": i_hits, "ii": ii_all}
        out["verdict"] = v
        print(f"verdict: (i) {'MET' if v['i'] else 'NOT MET'} ({i_hits} of {n}); (ii) {'MET' if v['ii'] else 'NOT MET'}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
