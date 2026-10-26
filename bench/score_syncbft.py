"""SYNC-BF16-T (docs/REGISTRY.md; judged only if SYNC-BF16-C's (i) is met): SmolLM2 GRPO with a bf16 merge-sync and
truncated importance sampling (train/grpo_lora8.py --tis 2), seeds 6, 7, 8, against SYNC-BF16-C's exact-sync and plain
merge-sync runs at the same seeds.
  (i)  the corrected merge-sync's final GSM8K accuracy is within 5 points of the exact-sync twin's in >= 2 of 3 seeds
  (ii) it beats the plain merge-sync at the same seed (McNemar p <= 0.05, corrected ahead) in >= 2 of 3 seeds
Descriptive: mismatch over steps 1-100, delivered share (syncs 1-10, 191-200), rewards, collapse step.
Files: logs/lb_syncbft_smol_gsm8k_bft_bf16merge_s<k>.out, results/std_bench/smol_gsm8k_bft_bf16merge_s<k>/bf16_exact/
gsm8k.jsonl; the twins as bench/score_syncbfc.py reads them.
    python3 bench/score_syncbft.py [--json results/std_bench/syncbft_verdict.json]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_syncbfc as c  # noqa: E402


def tlog(s: int) -> list[dict]:
    f = c.LB / f"logs/lb_syncbft_smol_gsm8k_bft_bf16merge_s{s}.out"
    return [json.loads(x) for x in f.read_text().splitlines() if x.startswith('{"step"')] if f.exists() else []


def tacc(s: int) -> dict | None:
    f = c.RES / f"smol_gsm8k_bft_bf16merge_s{s}/bf16_exact/gsm8k.jsonl"
    return {r["id"]: int(r["correct"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[6, 7, 8])
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cv = c.RES / "syncbfc_verdict.json"
    gate = json.loads(cv.read_text()).get("verdict", {}).get("i") if cv.exists() else None
    print(f"gate: SYNC-BF16-C (i) {'MET' if gate else 'NOT MET or not scored'}")
    out, i_hits, ii_hits, n = {"gate": gate}, 0, 0, 0
    for s in a.seeds:
        t, at, ae, am = tlog(s), tacc(s), c.acc("exact", s), c.acc("bf16merge", s)
        if not t or t[-1]["step"] < 200 or at is None or ae is None or am is None:
            print(f"seed {s}: not complete")
            continue
        n += 1
        ids = sorted(set(at) & set(ae) & set(am))
        pt, pe, pm = (100 * sum(d[i] for i in ids) / len(ids) for d in (at, ae, am))
        w, l, p = c.rr.mcnemar([at[i] for i in ids], [am[i] for i in ids])   # (corrected wins, plain wins, p)
        hit_i = abs(pt - pe) <= 5
        hit_ii = w > l and p <= 0.05
        i_hits += hit_i
        ii_hits += hit_ii
        out[str(s)] = {"acc_tis": pt, "acc_exact": pe, "acc_plain_merge": pm, "i": hit_i, "tis_vs_plain": [w, l, p],
                       "ii": hit_ii, "mismatch_1_100": c.mean([r.get("mismatch") for r in t if r["step"] <= 100]),
                       "delivered": [c.mean([r["sync_gain"] for r in t if r["step"] <= 10]),
                                     c.mean([r["sync_gain"] for r in t if r["step"] > 190])],
                       "reward_1_10": c.mean([r["reward"] for r in t if r["step"] <= 10]),
                       "reward_191_200": c.mean([r["reward"] for r in t if r["step"] > 190]),
                       "collapse_step": c.collapse_step(t)}
        print(f"seed {s}: GSM8K TIS merge {pt:.1f} vs exact {pe:.1f} vs plain merge {pm:.1f} (TIS vs plain {w}:{l}, p {p:.2g}); "
              f"mismatch 1-100 {out[str(s)]['mismatch_1_100']:.4f}; delivered {out[str(s)]['delivered'][0]:.3f} -> "
              f"{out[str(s)]['delivered'][1]:.3f}; reward {out[str(s)]['reward_1_10']:.2f} -> {out[str(s)]['reward_191_200']:.2f}; "
              f"collapse {out[str(s)]['collapse_step']}")
    if n == len(a.seeds) and gate:
        v = {"i": i_hits >= 2, "i_hits": i_hits, "ii": ii_hits >= 2, "ii_hits": ii_hits}
        out["verdict"] = v
        print(f"verdict: (i) {'MET' if v['i'] else 'NOT MET'} ({i_hits} of {n}); (ii) {'MET' if v['ii'] else 'NOT MET'} "
              f"({ii_hits} of {n})")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
